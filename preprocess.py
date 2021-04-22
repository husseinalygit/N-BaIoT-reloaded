import pandas as pd
import os
import time
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn', to stop SettingWithCopyWarning

start_time = time.time()

dataset_loc = 'dataset'
output_loc = 'processed'
device_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
normal_class = ['benign']
attack_classes = ['gafgyt.combo', 'gafgyt.junk', 'gafgyt.scan', 'gafgyt.tcp',
                  'gafgyt.udp', 'mirai.ack', 'mirai.scan', 'mirai.syn',
                  'mirai.udp', 'mirai.udpplain', ]

classes_dictionary = {normal_class[0]: 0, attack_classes[0]: 1, attack_classes[1]: 2, attack_classes[2]: 3,
                      attack_classes[3]: 4, attack_classes[4]: 5, attack_classes[5]: 6, attack_classes[6]: 7,
                      attack_classes[7]: 8, attack_classes[8]: 9, attack_classes[9]: 10}

for did in device_ids:
    print('started on device {}'.format(did))
    print('loading normal data')
    temp = pd.read_csv('{}/{}.{}.csv'.format(dataset_loc, did, normal_class[0]))
    training = temp[: int(len(temp) * 0.75)]
    testing_normal = temp.loc[int(len(temp) * 0.75):]
    testing_normal['class'] = classes_dictionary[normal_class[0]]
    testing_malicious = None
    for i, malicious_class in enumerate(attack_classes):
        print('loading attack {}'.format(malicious_class))
        try:
            malicious_data = pd.read_csv('{}/{}.{}.csv'.format(dataset_loc, did, malicious_class))
            malicious_data['class'] = classes_dictionary[malicious_class]
            if testing_malicious is None:
                testing_malicious = malicious_data
            else:
                testing_malicious = testing_malicious.append(malicious_data, ignore_index=True)
        except Exception as e:
            print(e)

    testing = testing_normal.append(testing_malicious, ignore_index=True)

    if not os.path.isdir('{}/{}'.format(output_loc, did)):
        os.makedirs('{}/{}'.format(output_loc, did))
    print('writing data for device {}'.format(did))
    testing.to_csv('{}/{}/{}'.format(output_loc, did, 'testing.csv'))
    training.to_csv('{}/{}/{}'.format(output_loc, did, 'training.csv'))

    print('Done on device {}'.format(did))

print("Execution took {} minutes".format(np.round((time.time() - start_time) / 60), 2))
