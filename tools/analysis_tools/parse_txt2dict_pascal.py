""" Parsing the result from text to dictionary.

Example:
    python parse_txt2dict.py ${test_robustness_result.txt} ${config_file.py}
"""

import sys
import numpy as np


def get_minimal_dictionary_class(dictionary):
    minimal_dictionary = {}
    minimal_dictionary['mAP'] = []
    minimal_dictionary['mPC'] = []
    minimal_dictionary['mAP_detail'] = {}
    minimal_dictionary['mPC_detail'] = {}

    # Classes
    classes = []
    for class_name in dictionary['gaussian_noise']['severity0'].keys():
        classes.append(class_name)
        if class_name == 'mAP':
            continue
        minimal_dictionary['mAP_detail'][class_name] = []
        minimal_dictionary['mPC_detail'][class_name] = []

    for corruption, severities  in dictionary.items():
        for severity, results in severities.items():
            # severity = ['severity0', 'severity1', ..., 'severity5']
            for class_name, result in results.items():
                # class_name = ['aeroplane', 'bicycle', ..., 'mAP']
                condition = 'mAP' if severity == 'severity0' else 'mPC'
                if class_name == 'mAP':
                    minimal_dictionary[f'{condition}'].append(float(result))
                else:
                    minimal_dictionary[f'{condition}_detail'][class_name].append(float(result['ap']))

    for condition in ['mAP', 'mPC']:
        minimal_dictionary[condition] = np.mean(minimal_dictionary[condition])
        for class_name in classes:
            if class_name == 'mAP':
                continue
            minimal_dictionary[f'{condition}_detail'][class_name] = np.mean(minimal_dictionary[f'{condition}_detail'][class_name])

    for condition in ['mAP', 'mPC']:
        print(f"=== {'clean result' if condition == 'mAP' else 'corrupted result'} ===")
        print(f"{condition}: {minimal_dictionary[condition]*100:.2f}")
        for class_name, result in minimal_dictionary[f'{condition}_detail'].items():
            print(f" > {class_name} ({result*100:.2f})")
        print("")

    return minimal_dictionary


def get_minimal_dictionary_corruption(dictionary):
    minimal_dictionary = {}

    # Corruption types
    corruptions = []
    for corruption in dictionary.keys():
        corruptions.append(corruption)
        minimal_dictionary[corruption] = []
    minimal_dictionary['clean'] = []

    for corruption, severities  in dictionary.items():
        if corruption == 'gaussian_noise':
            minimal_dictionary['clean'].append(float(severities['severity0']['mAP']))
        for severity, results in severities.items():
            if severity == 'severity0':
                continue
            minimal_dictionary[corruption].append(float(results['mAP']))

    for key, value in minimal_dictionary.items():
        minimal_dictionary[key] = np.mean(value)

    return minimal_dictionary


def get_dictionary(file_path):

    dictionary = {}

    with open(file_path) as file:
        for line in file:
            '''Corruption Type & Severity'''
            if line.startswith('Testing '):  # e.g., "Testing gaussian_noise at severity 0"
                corruption_type = line.split()[1]
                severity = int(line.split()[4])
                if not corruption_type in dictionary:
                    dictionary[corruption_type] = {}
                if not "severity" + str(severity) in dictionary[corruption_type]:
                    dictionary[corruption_type]["severity" + str(severity)] = {}

                for line in file:
                    '''time'''
                    if line.startswith('|'):
                        _line = file.readline()
                        while True:
                            _line = file.readline()
                            if _line.startswith('+'):
                                _line = file.readline()
                                _items = _line.replace(' ', '').split('|')
                                mAP = _items[-2]
                                dictionary[corruption_type]["severity" + str(severity)]['mAP'] = mAP
                                break
                            _items = _line.replace(' ', '').split('|')[1:]
                            class_name = _items[0]
                            gts = _items[1]
                            dets = _items[2]
                            recall = _items[3]
                            ap = _items[4]
                            dictionary[corruption_type]["severity" + str(severity)][class_name] = {
                                'gts': gts, 'dets': dets, 'recall': recall, 'ap': ap
                            }
                        break

    return dictionary

from mmcv import Config

def print_config_information(file_path):
    def print_dict(dict, name):
        print(f'  - {name}: (', end='')
        for key, value in dict.items():
            print(f'{key}={value}, ', end='')
        print(f')')
    cfg = Config.fromfile(file_path)

    ''' Model'''
    model = cfg.model
    print(f'=== config information ===')
    print(f'[model]')
    print_dict(model.rpn_head.loss_cls, 'rpn_cls')
    print_dict(model.rpn_head.loss_bbox, 'rpn_bbox')
    print_dict(model.roi_head.bbox_head.loss_cls, 'roi_cls')
    print_dict(model.roi_head.bbox_head.loss_bbox, 'roi_bbox')

    print(f'[data]')
    data = cfg.data
    print(f'  - samples_per_gpu={data.samples_per_gpu}, workers_per_gpu={data.workers_per_gpu}')
    train_pipeline = data.train.dataset.pipeline
    for i in range(len(train_pipeline)):
        if train_pipeline[i].type == 'AugMix':
            print_dict(train_pipeline[i], f'data.train.dataset.pipeline[{i}]')

    print(f'[runtime]')
    print_dict(cfg.evaluation, 'evaluation')
    print_dict(cfg.optimizer, 'optimizer')
    print(f'==========================')

def main():
    txt_file_path = sys.argv[1]
    config_file_path = sys.argv[2]
    if len(sys.argv) < 2:
        print("Insufficient arguments")
        sys.exit()
    print('txt file path : ' + txt_file_path)
    print('config file path : ' + config_file_path)

    dictionary = get_dictionary(txt_file_path)

    minimal_dictionary_class = get_minimal_dictionary_class(dictionary)
    print(f"mAP: {minimal_dictionary_class['mAP']:.2f}, mPC: {minimal_dictionary_class['mPC']:.2f}")
    minimal_dictionary_corruption = get_minimal_dictionary_corruption(dictionary)
    for key in minimal_dictionary_corruption.keys():
        print('key:', key, ' value:', minimal_dictionary_corruption[key] * 100)

if __name__ == '__main__':
    main()
