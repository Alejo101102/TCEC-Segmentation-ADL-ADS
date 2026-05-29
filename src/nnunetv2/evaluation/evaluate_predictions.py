import multiprocessing
import os
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
from scipy.ndimage import label as cc_label
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import (
    determine_reader_writer_from_dataset_json,
    determine_reader_writer_from_file_ending,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


# ---------------------------------------------------------------------------
# Umbrales de tamaño (mm³ = voxels con spacing 1x1x1)
# ---------------------------------------------------------------------------
SIZE_SMALL_MAX  = 100    # < 100 mm³       → small
SIZE_MEDIUM_MAX = 1000   # 100 – 1000 mm³  → medium
                         # > 1000 mm³      → large


def get_lesion_size_category(volume_voxels: int) -> str:
    if volume_voxels < SIZE_SMALL_MAX:
        return 'small'
    elif volume_voxels < SIZE_MEDIUM_MAX:
        return 'medium'
    else:
        return 'large'


# ---------------------------------------------------------------------------
# Helpers heredados (sin cambios respecto al original)
# ---------------------------------------------------------------------------

def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '').replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    results_converted = deepcopy(results)
    results_converted['mean'] = {
        label_or_region_to_key(k): results['mean'][k] for k in results['mean']
    }
    for i in range(len(results_converted['metric_per_case'])):
        results_converted['metric_per_case'][i]['metrics'] = {
            label_or_region_to_key(k): results['metric_per_case'][i]['metrics'][k]
            for k in results['metric_per_case'][i]['metrics']
        }
        results_converted['metric_per_case'][i]['lesion_metrics'] = {
            label_or_region_to_key(k): results['metric_per_case'][i]['lesion_metrics'][k]
            for k in results['metric_per_case'][i]['lesion_metrics']
        }
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean']}
    for i in range(len(results['metric_per_case'])):
        results['metric_per_case'][i]['metrics'] = {
            key_to_label_or_region(k): results['metric_per_case'][i]['metrics'][k]
            for k in results['metric_per_case'][i]['metrics']
        }
        results['metric_per_case'][i]['lesion_metrics'] = {
            key_to_label_or_region(k): results['metric_per_case'][i]['lesion_metrics'][k]
            for k in results['metric_per_case'][i]['lesion_metrics']
        }
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    mask = np.zeros_like(segmentation, dtype=bool)
    for r in region_or_label:
        mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray,
                         ignore_mask: np.ndarray = None):
    use_mask = np.ones_like(mask_ref, dtype=bool) if ignore_mask is None else ~ignore_mask
    tp = np.sum((mask_ref  &  mask_pred) & use_mask)
    fp = np.sum((~mask_ref &  mask_pred) & use_mask)
    fn = np.sum((mask_ref  & ~mask_pred) & use_mask)
    tn = np.sum((~mask_ref & ~mask_pred) & use_mask)
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Métricas para una lesión individual
# ---------------------------------------------------------------------------

def compute_lesion_metrics_single(mask_ref: np.ndarray,
                                   mask_pred: np.ndarray) -> dict:
    tp = int(np.sum( mask_ref &  mask_pred))
    fp = int(np.sum(~mask_ref &  mask_pred))
    fn = int(np.sum( mask_ref & ~mask_pred))
    tn = int(np.sum(~mask_ref & ~mask_pred))

    if tp + fp + fn == 0:
        dice = float('nan')
        iou  = float('nan')
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou  = tp / (tp + fp + fn)

    return {
        'Dice':   dice,
        'IoU':    iou,
        'TP':     tp,
        'FP':     fp,
        'FN':     fn,
        'TN':     tn,
        'n_pred': fp + tp,
        'n_ref':  fn + tp,
    }


# ---------------------------------------------------------------------------
# Evaluación por lesión individual — un par de archivos
# ---------------------------------------------------------------------------

def compute_metrics_per_lesion(reference_file: str,
                                prediction_file: str,
                                image_reader_writer: BaseReaderWriter,
                                labels_or_regions,
                                ignore_label: int = None,
                                iou_threshold: float = 0.1) -> dict:
    seg_ref,  _ = image_reader_writer.read_seg(reference_file)
    seg_pred, _ = image_reader_writer.read_seg(prediction_file)

    ignore_mask = (seg_ref == ignore_label) if ignore_label is not None else None

    results = {
        'reference_file':  reference_file,
        'prediction_file': prediction_file,
        'metrics':         {},
        'lesion_metrics':  {},
    }

    for r in labels_or_regions:

        # ── Máscaras globales ────────────────────────────────────────────────
        mask_ref_global  = region_or_label_to_mask(seg_ref,  r)
        mask_pred_global = region_or_label_to_mask(seg_pred, r)

        if ignore_mask is not None:
            mask_ref_global  = mask_ref_global  & ~ignore_mask
            mask_pred_global = mask_pred_global & ~ignore_mask

        # ── Métricas globales por imagen (igual que el original) ─────────────
        tp_g, fp_g, fn_g, tn_g = compute_tp_fp_fn_tn(mask_ref_global, mask_pred_global)
        if tp_g + fp_g + fn_g == 0:
            results['metrics'][r] = {
                'Dice': float('nan'), 'IoU': float('nan'),
                'TP': int(tp_g), 'FP': int(fp_g),
                'FN': int(fn_g), 'TN': int(tn_g),
                'n_pred': int(fp_g + tp_g), 'n_ref': int(fn_g + tp_g),
            }
        else:
            results['metrics'][r] = {
                'Dice': 2 * tp_g / (2 * tp_g + fp_g + fn_g),
                'IoU':  tp_g / (tp_g + fp_g + fn_g),
                'TP': int(tp_g), 'FP': int(fp_g),
                'FN': int(fn_g), 'TN': int(tn_g),
                'n_pred': int(fp_g + tp_g), 'n_ref': int(fn_g + tp_g),
            }

        struct = np.ones((3,) * seg_ref.ndim, dtype=bool)

        labeled_ref,  n_ref  = cc_label(mask_ref_global,  structure=struct)
        labeled_pred, n_pred = cc_label(mask_pred_global, structure=struct)

        per_lesion       = []
        matched_pred_ids = set()
        lesion_id        = 0

        for ref_id in range(1, n_ref + 1):
            lesion_id    += 1
            ref_component = labeled_ref == ref_id
            ref_vol       = int(np.sum(ref_component))

            best_iou     = 0.0
            best_pred_id = None

            for pred_id in range(1, n_pred + 1):
                pred_component = labeled_pred == pred_id
                intersection   = int(np.sum(ref_component & pred_component))
                if intersection == 0:
                    continue
                union   = int(np.sum(ref_component | pred_component))
                iou_val = intersection / union if union > 0 else 0.0
                if iou_val > best_iou:
                    best_iou     = iou_val
                    best_pred_id = pred_id

            if best_pred_id is not None and best_iou >= iou_threshold:
                pred_component = labeled_pred == best_pred_id
                pred_vol       = int(np.sum(pred_component))
                matched_pred_ids.add(best_pred_id)

                m = compute_lesion_metrics_single(ref_component, pred_component)
                m.update({
                    'lesion_id':         lesion_id,
                    'lesion_type':       'matched',
                    'size_category':     get_lesion_size_category(ref_vol),
                    'ref_volume_mm3':    ref_vol,
                    'pred_volume_mm3':   pred_vol,
                    'matching_IoU':      best_iou,
                    'ref_component_id':  ref_id,
                    'pred_component_id': best_pred_id,
                })
            else:
                m = compute_lesion_metrics_single(ref_component,
                                                   np.zeros_like(ref_component))
                m.update({
                    'lesion_id':         lesion_id,
                    'lesion_type':       'false_negative',
                    'size_category':     get_lesion_size_category(ref_vol),
                    'ref_volume_mm3':    ref_vol,
                    'pred_volume_mm3':   0,
                    'matching_IoU':      best_iou,
                    'ref_component_id':  ref_id,
                    'pred_component_id': None,
                })

            per_lesion.append(m)

        for pred_id in range(1, n_pred + 1):
            if pred_id in matched_pred_ids:
                continue
            lesion_id     += 1
            pred_component = labeled_pred == pred_id
            pred_vol       = int(np.sum(pred_component))

            m = compute_lesion_metrics_single(np.zeros_like(pred_component),
                                               pred_component)
            m.update({
                'lesion_id':         lesion_id,
                'lesion_type':       'false_positive',
                'size_category':     get_lesion_size_category(pred_vol),
                'ref_volume_mm3':    0,
                'pred_volume_mm3':   pred_vol,
                'matching_IoU':      0.0,
                'ref_component_id':  None,
                'pred_component_id': pred_id,
            })
            per_lesion.append(m)

        n_matched = len(matched_pred_ids)
        results['lesion_metrics'][r] = {
            'n_ref_lesions':         n_ref,
            'n_pred_lesions':        n_pred,
            'n_matched':             n_matched,
            'n_false_negative':      n_ref  - n_matched,
            'n_false_positive':      n_pred - n_matched,
            'lesion_detection_rate': (n_matched / n_ref  if n_ref  > 0 else float('nan')),
            'lesion_fp_rate':        ((n_pred - n_matched) / n_pred if n_pred > 0 else float('nan')),
            'per_lesion':            per_lesion,
        }

    return results


def compute_metrics_on_folder_per_lesion(
        folder_ref: str,
        folder_pred: str,
        output_file: str,
        image_reader_writer: BaseReaderWriter,
        file_ending: str,
        regions_or_labels,
        ignore_label: int = None,
        num_processes: int = default_num_processes,
        iou_threshold: float = 0.1,
        chill: bool = True) -> dict:
    
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file debe terminar en .json'

    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref  = subfiles(folder_ref,  suffix=file_ending, join=False)

    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "No todos los archivos de folder_ref existen en folder_pred"

    files_ref  = [join(folder_ref,  i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics_per_lesion,
            list(zip(
                files_ref, files_pred,
                [image_reader_writer] * len(files_pred),
                [regions_or_labels]   * len(files_pred),
                [ignore_label]        * len(files_pred),
                [iou_threshold]       * len(files_pred),
            ))
        )

    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())

    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = float(np.nanmean([i['metrics'][r][m] for i in results]))

    foreground_mean = {}
    for m in metric_list:
        values = [means[k][m] for k in means if k not in (0, '0')]
        foreground_mean[m] = float(np.nanmean(values)) if values else float('nan')

    size_categories = ['small', 'medium', 'large']
    size_ranges     = {'small': '<100 mm3', 'medium': '100-1000 mm3', 'large': '>1000 mm3'}

    lesion_summary = {}
    for r in regions_or_labels:

        total_ref = total_pred = total_matched = total_fn = total_fp = 0
        all_matched_dice = []

        by_size = {
            cat: {
                'n_ref': 0, 'n_matched': 0,
                'n_false_negative': 0, 'n_false_positive': 0,
                'dice_list': []
            }
            for cat in size_categories
        }

        for case in results:
            lm = case['lesion_metrics'][r]
            total_ref     += lm['n_ref_lesions']
            total_pred    += lm['n_pred_lesions']
            total_matched += lm['n_matched']
            total_fn      += lm['n_false_negative']
            total_fp      += lm['n_false_positive']

            for lesion in lm['per_lesion']:
                cat = lesion['size_category']
                if lesion['lesion_type'] == 'matched':
                    all_matched_dice.append(lesion['Dice'])
                    by_size[cat]['n_ref']     += 1
                    by_size[cat]['n_matched'] += 1
                    by_size[cat]['dice_list'].append(lesion['Dice'])
                elif lesion['lesion_type'] == 'false_negative':
                    by_size[cat]['n_ref']            += 1
                    by_size[cat]['n_false_negative'] += 1
                elif lesion['lesion_type'] == 'false_positive':
                    by_size[cat]['n_false_positive'] += 1

        by_size_summary = {}
        for cat in size_categories:
            d         = by_size[cat]
            n_ref     = d['n_ref']
            n_matched = d['n_matched']
            dice_list = d['dice_list']
            by_size_summary[cat] = {
                'size_range':            size_ranges[cat],
                'n_ref_lesions':         n_ref,
                'n_matched':             n_matched,
                'n_false_negative':      d['n_false_negative'],
                'n_false_positive':      d['n_false_positive'],
                'lesion_detection_rate': (n_matched / n_ref
                                          if n_ref > 0 else float('nan')),
                'mean_Dice_matched':     (float(np.nanmean(dice_list))
                                          if dice_list else float('nan')),
                'std_Dice_matched':      (float(np.nanstd(dice_list))
                                          if dice_list else float('nan')),
            }

        lesion_summary[label_or_region_to_key(r)] = {
            'total_ref_lesions':         total_ref,
            'total_pred_lesions':        total_pred,
            'total_matched':             total_matched,
            'total_false_negatives':     total_fn,
            'total_false_positives':     total_fp,
            'lesion_detection_rate':     (total_matched / total_ref
                                          if total_ref > 0 else float('nan')),
            'mean_Dice_matched_lesions': (float(np.nanmean(all_matched_dice))
                                          if all_matched_dice else float('nan')),
            'std_Dice_matched_lesions':  (float(np.nanstd(all_matched_dice))
                                          if all_matched_dice else float('nan')),
            'by_size_category':          by_size_summary,
        }

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    recursive_fix_for_json_export(lesion_summary)

    means_str = {label_or_region_to_key(k): v for k, v in means.items()}

    for case in results:
        case['lesion_metrics'] = {
            label_or_region_to_key(k): v for k, v in case['lesion_metrics'].items()
        }
        case['metrics'] = {
            label_or_region_to_key(k): v for k, v in case['metrics'].items()
        }

    result = {
        'foreground_mean': foreground_mean,
        'mean':            means_str,
        'lesion_summary':  lesion_summary,
        'metric_per_case': results,
    }

    if output_file is not None:
        save_json(result, output_file, sort_keys=True)

    return result

def compute_metrics_on_folder(
        folder_ref: str,
        folder_pred: str,
        output_file: str,
        image_reader_writer,
        file_ending: str,
        regions_or_labels,
        ignore_label: int = None,
        num_processes: int = default_num_processes,
        chill: bool = True):

    return compute_metrics_on_folder_per_lesion(
        folder_ref=folder_ref,
        folder_pred=folder_pred,
        output_file=output_file,
        image_reader_writer=image_reader_writer,
        file_ending=file_ending,
        regions_or_labels=regions_or_labels,
        ignore_label=ignore_label,
        num_processes=num_processes,
        chill=chill
    )


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    file_ending  = dataset_json['file_ending']
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()
    if output_file is None:
        output_file = join(folder_pred, 'summary_per_lesion.json')
    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder_per_lesion(
        folder_ref, folder_pred, output_file, rw, file_ending,
        lm.foreground_regions if lm.has_regions else lm.foreground_labels,
        lm.ignore_label, num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str,
                                     labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending  = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(
        file_ending, example_file, allow_nonmatching_filename=True, verbose=False)()
    if output_file is None:
        output_file = join(folder_pred, 'summary_per_lesion.json')
    compute_metrics_on_folder_per_lesion(
        folder_ref, folder_pred, output_file, rw, file_ending,
        labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder',   type=str)
    parser.add_argument('pred_folder', type=str)
    parser.add_argument('-djfile', type=str, required=True)
    parser.add_argument('-pfile',  type=str, required=True)
    parser.add_argument('-o',  type=str, required=False, default=None)
    parser.add_argument('-np', type=int, required=False, default=default_num_processes)
    parser.add_argument('--chill', action='store_true')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile,
                               args.pfile, args.o, args.np, chill=args.chill)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder',   type=str)
    parser.add_argument('pred_folder', type=str)
    parser.add_argument('-l',  type=int, nargs='+', required=True)
    parser.add_argument('-il', type=int, required=False, default=None)
    parser.add_argument('-o',  type=str, required=False, default=None)
    parser.add_argument('-np', type=int, required=False, default=default_num_processes)
    parser.add_argument('--chill', action='store_true')
    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l,
                                     args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref  = '/home/castanojenifer/Codigos/labelsTs'
    folder_pred = '/home/castanojenifer/Codigos/output_SRI24'
    output_file = '/home/castanojenifer/Codigos/output_SRI24/summary_per_lesion.json'

    gt_files   = os.listdir(folder_ref)
    pred_files = os.listdir(folder_pred)
    print(f"Archivos encontrados en GT: {len(gt_files)}")
    print(f"Archivos encontrados en Predicciones: {len(pred_files)}")

    image_reader_writer = SimpleITKIO()
    file_ending         = '.nii.gz'
    regions             = labels_to_list_of_regions([1])
    ignore_label        = None
    num_processes       = 12
    iou_threshold       = 0.1

    compute_metrics_on_folder_per_lesion(
        folder_ref, folder_pred, output_file,
        image_reader_writer, file_ending,
        regions, ignore_label, num_processes, iou_threshold,
    )