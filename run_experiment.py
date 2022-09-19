import json
import os
import winsound

def run_experiment(data, id_list):
    for record in data:
        if record['ID'] in id_list:
            command = ("python test_batch.py --model {} --n_epochs {} --batch_size {}"
            " --learning_rate {} --calibration {} --output_path {} --ensemble {}")\
                .format(record['model'], record['n_epochs'], record['batch_size'],
                        record['learning_rate'], record['calibration'],
                        record['output_path'], record['ensemble'])
            print(command)
            os.system(command)
            print("Experiment ", record['ID'], " finished.")


if __name__ == "__main__":
    with open('experiment/configs.json') as json_file:
        data = json.load(json_file)

    data = data['test']
    id_list = [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    run_experiment(data, id_list)
    winsound.Beep(440, 2000)
