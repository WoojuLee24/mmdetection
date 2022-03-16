""" Parsing the result from text to dictionary.

Example:
    python parse_txt2dict.py ${RESULT_FILE_PATH}
"""

import sys
import re


def main():

    file_path = sys.argv[1]
    if len(sys.argv) != 2:
        print("Insufficient arguments")
        sys.exit()
    print('file path : ' + file_path)

    dictionary = {}
    keys = ['cleanP_all', 'cleanP_small', 'cleanP_medium', 'cleanP_large', 'corr_mPC_all', 'corr_mPC_small', 'corr_mPC_medium', 'corr_mPC_large'
                   , 'gaussian_noise', 'shot_noise', 'impulse_noise'
                   , 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
                   , 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    for key in keys:
        dictionary[key] = -1.0

    mpc = False
    with open(file_path) as file:
        for line in file:
            '''Corruption Type & Severity'''
            if line.startswith('Testing '): # e.g., "Testing gaussian_noise at severity 0"
                corruption_type = line.split()[1]
                severity = int(line.split()[4])

            '''Average Precision & Average Recall & Mean Performance under Corruption [mPC] (bbox)'''
            if line.startswith('Mean Performance under Corruption [mPC] (bbox)'):
                mpc = True

            if line.startswith(' Average Precision') or line.startswith(' Average Recall'):
                words = re.split('[=,|,\[,\],]', line.replace(' ', '').replace('\n', '')) # e.g., " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409"
                iou = words[2]
                area = words[4]
                max_dets = words[6]
                score = words[8]
                if words[0].startswith('AveragePrecision'):
                    score_type = 'AP'
                elif words[0].startswith('AverageRecall'):
                    score_type = 'AR'
                # print('   IoU=' + iou + '  area=' + area + '  maxDets=' + max_dets + '  '+score_type+'=' + score)

                if iou == '0.50:0.95' and max_dets == '100' and score_type == 'AP':
                    if mpc:
                        # 'corr_mPC_all', 'corr_mPC_small', 'corr_mPC_medium', 'corr_mPC_large'
                        dictionary['corr_mPC_' + area] = float(score)
                    elif 0 < severity and area == 'all':
                        # 'gaussian_noise', 'shot_noise', 'impulse_noise',
                        # 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                        # 'snow', 'frost', 'fog', 'brightness', 'contrast',
                        # 'elastic_transform', 'pixelate', 'jpeg_compression'
                        if severity == 1:
                            scores = 0
                        scores = scores+float(score)
                        if severity == 5:
                            dictionary[corruption_type] = scores/5
                    elif severity == 0:
                        # 'cleanP_all', 'cleanP_small', 'cleanP_medium', 'cleanP_large'
                        str_tmp = "cleanP_"+area
                        dictionary[str_tmp] = float(score)

    for key in dictionary.keys():
        print('key:', key, ' value:', dictionary[key]*100)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

