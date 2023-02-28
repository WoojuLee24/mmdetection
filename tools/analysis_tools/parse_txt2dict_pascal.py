""" Parsing the result from text to dictionary.

Example:
    python parse_txt2dict.py ${test_robustness_result.txt} ${config_file.py}
"""

import sys
import numpy as np


def get_minimal_dictionary(dictionary):
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

            '''time'''
            if 'class' in line and 'gts' in line:
            # if line.startswith('| class       | gts  | dets  | recall | ap    |'):
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
    minimal_dictionary = get_minimal_dictionary(dictionary)
    for key in minimal_dictionary.keys():
        if isinstance(minimal_dictionary[key], dict):
            for k, v in minimal_dictionary[key].items():
                print('key:', k, ' value:', v * 100)
        else:
            print('key:', key, 'value:', minimal_dictionary[key] * 100)

if __name__ == '__main__':
    main()
