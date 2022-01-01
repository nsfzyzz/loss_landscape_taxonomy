import argparse
import numpy as np
import os


def get_script(args, BASH_COMMAND_LIST):
    
    print("Start writing the command list!")
    
    info = """"""
        
    for command in BASH_COMMAND_LIST:

        if args.slurm_commands:
            info += f"srun -N 1 -n 1 --gres=gpu:1 {command} & \n \n"
        else:
            info += f"{command} \n"
        
        #info += f"job_limit {args.num_gpus} \n \n"

    script = info
    
    if os.path.isfile(args.file_name):
        with open(args.file_name, 'w') as rsh:
            rsh.truncate()

    with open (args.file_name, 'w') as rsh:
        rsh.write(script)
        
    os.system(f"chmod +x {args.file_name}")
                
            
def get_load_temperature(args):
    
    ## Different load
    if args.load_range == 'coarse':
        load = [(0.0, '0'), (0.1, '01'), (0.2, '02'), (0.3, '03'), (0.4, '04'), (0.5, '05'), (0.6, '06'), (0.7, '07'), (0.8, '08')]
    elif args.load_range == 'divide_coarse':
        load = [(0.2, '02'), (0.3, '03'), (0.4, '04'), (0.5, '05'), (0.6, '06'), (0.7, '07'), (0.8, '08'), (0.9, '09'), (1.0, '10')]
    elif args.load_range == 'subset_coarse':
        load = [(0.1, '01'), (0.2, '02'), (0.3, '03'), (0.4, '04'), (0.5, '05'), (0.6, '06'), (0.7, '07'), (0.8, '08'), (0.9, '09'), (1.0, '10')]
    elif args.load_range == 'different_width_coarse':
        load = [(0.1, '01'), (0.2, '02'), (0.3, '03'), (0.4, '04'), (0.5, '05'), (0.6, '06'), (0.7, '07'), (0.8, '08')]
    elif args.load_range == 'fine':
        load = [(0.0, '0'), (0.025, '0025'), (0.05, '005'), (0.075, '0075'), (0.1, '01'), (0.15, '015'), (0.2, '02'), (0.25, '025'), (0.3, '03'), (0.35, '035'), (0.4, '04'), (0.45, '045'), (0.5, '05'), (0.55, '055'), (0.6, '06'), (0.65, '065'), (0.7, '07'), (0.75, '075'), (0.8, '08')]
    elif args.load_range == 'only_fine':
        load = [(0.025, '0025'), (0.05, '005'), (0.075, '0075'), (0.15, '015'), (0.25, '025'), (0.35, '035'), (0.45, '045'), (0.55, '055'), (0.65, '065'), (0.75, '075')]
    elif args.load_range == 'divide_fine':
        load = [(0.2, '02'), (0.3, '03'), (0.35, '035'), (0.4, '04'), (0.45, '045'), (0.5, '05'), (0.55, '055'), (0.6, '06'), (0.65, '065'), (0.7, '07'), (0.75, '075'), (0.8, '08'), (0.85, '085'), (0.9, '09'), (0.95, '095'), (1.0, '10')]
    elif args.load_range == 'constant':
        load = [(1.0, '10'), (1.0, '10'), (1.0, '10'), (1.0, '10'), (1.0, '10'), (1.0, '10'), 
               (1.0, '10'), (1.0, '10'), (1.0, '10'), (1.0, '10'), (1.0, '10'), (1.0, '10')]
    elif args.load_range == 'constant_small':
        load = [(0.1, '01'), (0.1, '01'), (0.1, '01'), (0.1, '01'), (0.1, '01'), (0.1, '01'), 
               (0.1, '01'), (0.1, '01'), (0.1, '01'), (0.1, '01'), (0.1, '01'), (0.1, '01')]
    elif args.load_range == 'customize':
        load_map = {'0': (0.0, '0'), '0025': (0.025, '0025'), '005': (0.05, '005'), '0075': (0.075, '0075'), '01': (0.1, '01'), '0125': (0.125, '0125'), '015': (0.15, '015'), '0175': (0.175, '0175'), '02': (0.2, '02'), '025': (0.25, '025'), '03': (0.3, '03'), '035': (0.35, '035'), '04': (0.4, '04'), '045': (0.45, '045'), '05': (0.5, '05'), '055': (0.55, '055'), '06': (0.6, '06'), '065': (0.65, '065'), '07': (0.7, '07'), '075': (0.75, '075'), '08': (0.8, '08'), '085': (0.85, '085'), '09': (0.9, '09'), '095': (0.95, '095'), '10': (1.0, '10')}
        
        load = [load_map[load] for load in args.load]
    
    ## Different temperature
    if args.temperature_range == 'customize':
        temperature = args.temperature
        return load, temperature
    elif args.temperature_range == 'coarse':
        lr_list = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]
        bs_list = [16, 32, 64, 128, 256, 512, 1024]
        wd_list = [3.2e-3, 1.6e-3, 8e-4, 4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5]
    elif args.temperature_range == 'fine':
        lr_list = [0.4, 0.28, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.0177, 0.0125, 0.0088, 0.00625, 0.0044, 0.003125, 0.0022, 0.0015625]
        bs_list = [16, 17, 19, 21, 24, 27, 32, 44, 64, 92, 128, 180, 256, 364, 512, 724, 1024]
        wd_list = [5e-3, 4.5e-3, 4e-3, 3.5e-3, 3e-3, 2.5e-3, 2e-3, 1.5e-3, 1e-3, 7e-4, 5e-4, 3.2e-4, 2e-4, 1.4e-4, 1e-4, 7e-5, 5e-5]
    elif args.temperature_range == 'only_fine':
        bs_list = [17, 19, 21, 24, 27, 44, 92, 180, 364, 724]
    elif args.temperature_range == 'Efficient_fine':
        bs_list = [16, 24, 32, 44, 64, 92, 128, 180, 256, 364, 512, 724, 1024]
    
    if args.temperature_type == 'bs':
        temperature = bs_list
    elif args.temperature_type == 'lr':
        temperature = lr_list
    elif args.temperature_type == 'wd':
        temperature = wd_list
    
    return load, temperature

    
def get_training_params(args, val):
    
    lr = args.lr_standard
    bs = args.bs_standard
    wd = args.wd_standard
    
    if args.temperature_type == 'lr':
        lr = val
    elif args.temperature_type == 'bs':
        bs = val
    elif args.temperature_type == 'wd':
        wd = val
        
    if args.training_type == 'small_wd':
        wd = 0.0001
    if args.training_type == 'lr_scaling':
        lr = args.lr_standard * (bs/args.bs_standard)
        
    return lr, bs, wd


def get_ckpt_folder(args, home_dir, val, load_suffix):
    
    ckpt_folder = os.path.join(home_dir, 
                "checkpoint/different_knobs_{5}_{4}/{0}_{1}/{2}/{3}/".format(args.temperature_type, val, args.training_type, args.arch, load_suffix, args.folder_suffix))
    
    return ckpt_folder


def get_curve_folder(args, home_dir, val, load_suffix, exp_id_1=0, exp_id_2=1):
        
    curve_folder = os.path.join(home_dir, "checkpoint/curve_fitting_different_knobs_{7}_{6}/{0}_{1}/{2}/{3}/connecting_{4}_{5}/".format(args.temperature_type, val, args.training_type, args.arch, exp_id_1, exp_id_2, load_suffix, args.folder_suffix))
    
    return curve_folder


def get_two_check_points(args, ckpt_folder, exp_id_1=0, exp_id_2=1):
    
    checkpoint1 = os.path.join(ckpt_folder, f'net_exp_{exp_id_1}.pkl')
    checkpoint2 = os.path.join(ckpt_folder, f'net_exp_{exp_id_2}.pkl')
    early_stopped_checkpoint1 = os.path.join(ckpt_folder, f'net_exp_{exp_id_1}_early_stopped_model.pkl')
    early_stopped_checkpoint2 = os.path.join(ckpt_folder, f'net_exp_{exp_id_2}_early_stopped_model.pkl')
    best_checkpoint1 = os.path.join(ckpt_folder, f'net_exp_{exp_id_1}_best.pkl')
    best_checkpoint2 = os.path.join(ckpt_folder, f'net_exp_{exp_id_2}_best.pkl')

    if args.early_stop_checkpoint:
        if os.path.exists(early_stopped_checkpoint1):
            checkpoint1 = early_stopped_checkpoint1
        elif os.path.exists(best_checkpoint1):
            checkpoint1 = best_checkpoint1

        if os.path.exists(early_stopped_checkpoint2):
            checkpoint2 = early_stopped_checkpoint2
        elif os.path.exists(best_checkpoint2):
            checkpoint2 = best_checkpoint2

    return checkpoint1, checkpoint2


def create_metric_folder(args, folder='.', name='result.pkl'):
    
    result_folder_name = os.path.join(folder, 'metrics/')
    result_name = os.path.join(result_folder_name, f'{name}')

    if args.create_folder:
        if not os.path.exists(result_folder_name):
            os.makedirs(result_folder_name)
            
    return result_folder_name, result_name


def compare_temperature_threshold(args, val):
    
    if args.temperature_type == 'bs':
        return val <= args.efficient_threshold
    if args.temperature_type == 'lr':
        return val >= args.efficient_threshold
    if args.temperature_type == 'wd':
        return val >= args.efficient_threshold
    

def get_command_list(args):
    
    BASH_COMMAND_LIST = []    

    load_list, temperature_list = get_load_temperature(args)
    
    home_dir = '../'
    
    for load, load_suffix in load_list:
        
        for val in temperature_list:
            
            # Get the folder names
            ckpt_folder = get_ckpt_folder(args, home_dir, val, load_suffix)
            random_label_path = f'../../data/random_labels/random_label_{load_suffix}_normal.pkl'
            random_label_path_test = f'../../data/random_labels/random_label_{load_suffix}_normal_test.pkl'
            
            if args.subset_noisy_prob == 0.1:
                random_label_path_subset_noisy = f'../../data/random_labels/random_label_01_normal.pkl'
                random_label_path_test_subset_noisy = f'../../data/random_labels/random_label_01_normal_test.pkl'
            else:
                raise NameError('subset noisy value not implemented yet!')
                        
            # Some common commands
            command_suffix = ''
            if args.different_width:
                command_suffix += f' --different-width --resnet18-width {args.width}'
            
            data_command = ''
            if args.data_type == 'random_label':
                data_command = f"--random-labels --label-corrupt-prob {load}" \
                               + f" --random-label-path {random_label_path} --shuffle-random-data"
                if args.train_or_test == "test" and args.test_on_noise:
                    data_command += f" --random-label-path-test {random_label_path_test}"
                    
            elif args.data_type == 'subset' or args.data_type == 'subset_augmentation':
                data_command = f"--data-subset --subset {load}"
                if args.data_type == 'subset_augmentation' and not args.not_augment_CKA_for_data_augmentation_exp:
                    data_command += " --augmentation-subset"
                
            elif args.data_type == 'subset_noisy':
                data_command = f"--data-subset --subset {load} --subset-noisy "
                data_command += f"--label-corrupt-prob {args.subset_noisy_prob}" \
                               + f" --random-label-path {random_label_path_subset_noisy} --shuffle-random-data"
                if args.train_or_test == "test" and args.test_on_noise:
                    data_command += f" --random-label-path-test {random_label_path_test_subset_noisy}"
                
            if args.test_on_noise:
                data_command += " --test-on-noise"
                
            
            # Two different modes if using training or testing data
            if args.experiment_type in ['hessian', 'CKA', 'curve', 'loss_acc']:
                data_command += f" --train-or-test {args.train_or_test}"
                train_or_test_suffix = ""
                if args.train_or_test == 'test':
                    train_or_test_suffix = "_test" 
                    if args.test_on_noise:
                        train_or_test_suffix += "_on_noise"
            
            # experiments for different metrics
            if args.experiment_type == 'dist':
                
                if args.early_stop_checkpoint:
                    command_suffix += ' --early-stopping'
                    
                if args.data_type == 'cifar100':
                    command_suffix  += ' --cifar100'
                
                result_folder_name, result_name = create_metric_folder(args, folder=ckpt_folder, name='model_dist.pkl')

                command = f"python ./code/model_pair_distance.py --arch {args.arch}{command_suffix} " \
                          + f"--checkpoint-folder {ckpt_folder} --result-location {result_name} " \
                          + f"1>{result_folder_name}/model_dist.log " \
                          + f"2>{result_folder_name}/model_dist.err"
                
                BASH_COMMAND_LIST.append((command, result_name))
            
            if args.experiment_type == 'CKA':
                
                if args.early_stop_checkpoint:
                    command_suffix += ' --early-stopping'
                    
                if args.mixup_CKA:
                    command_suffix += f' --mixup-CKA --mixup-alpha {args.mixup_alpha}'
                    train_or_test_suffix = f"_mixup_alpha_{args.mixup_alpha}"
                    
                if args.CKA_batches != 5:
                    command_suffix += f' --mixup-CKA --mixup-alpha {args.mixup_alpha} --CKA-batches {args.CKA_batches}'
                    train_or_test_suffix = f"_mixup_alpha_{args.mixup_alpha}_batch_{args.CKA_batches}"
                    
                if args.CKA_repeat != 1:
                    command_suffix += f' --mixup-CKA --mixup-alpha {args.mixup_alpha} --CKA-repeat-runs {args.CKA_repeat}'
                    train_or_test_suffix = f"_mixup_alpha_{args.mixup_alpha}_repeat_{args.CKA_repeat}"
                    
                # Here, we use the output-based CKA to save time
                command_suffix += ' --not-input'
                                                        
                result_folder_name, result_name = create_metric_folder(args, folder=ckpt_folder, name=f'CKA{train_or_test_suffix}.pkl')

                command = f"python ./code/CKA.py --training-type {args.training_type} " \
                          + f"--arch {args.arch} " \
                          + f"{data_command} --checkpoint-folder {ckpt_folder}{command_suffix} " \
                          + f"--result-location {result_name} " \
                          + f"1>{result_folder_name}/CKA{train_or_test_suffix}.log " \
                          + f"2>{result_folder_name}/CKA{train_or_test_suffix}.err"

                BASH_COMMAND_LIST.append((command, result_name))
                                
            if args.experiment_type == 'hessian':
                
                if args.early_stop_checkpoint:
                    command_suffix += ' --early-stopping'
                
                if args.exp_num!=5:
                    command_suffix += f' --exp-num {args.exp_num}'
                    
                # This code should not matter if we use noisy labels.
                hessian_batch_size = args.hessian_batch * args.mini_hessian_batch_size
                
                result_folder_name, result_name = create_metric_folder(args, folder=ckpt_folder, name=f'hessian{train_or_test_suffix}.pkl')
                
                command = f"python ./code/measure_hessian.py --training-type {args.training_type} " \
                         + f"--arch {args.arch} {data_command} " \
                         + f"--hessian-batch-size {hessian_batch_size} " \
                         + f"--mini-hessian-batch-size {args.mini_hessian_batch_size} " \
                         + f"--checkpoint-folder {ckpt_folder}{command_suffix} --result-location {result_name} " \
                         + f"1>{result_folder_name}/hessian{train_or_test_suffix}.log " \
                         + f"2>{result_folder_name}/hessian{train_or_test_suffix}.err"
                
                BASH_COMMAND_LIST.append((command, result_name))
                
            elif args.experiment_type == 'curve':
                
                if args.multiple_curves:
                    exps = [(0,1), (1,2), (2,3), (3,4), (4,0)]
                else:
                    exps = [(0,1)]
                
                for exp_id_1,exp_id_2 in exps:
                
                    curve_folder = get_curve_folder(args, home_dir, val, load_suffix, exp_id_1=exp_id_1, exp_id_2=exp_id_2)

                    if args.create_folder:
                        if not os.path.exists(curve_folder):
                            os.makedirs(curve_folder)

                    result_name = f'curve{args.result_suffix}.npz'
                    result_name = os.path.join(curve_folder, result_name)
                    log_name = f'{curve_folder}/curve{args.result_suffix}'

                    checkpoint1, checkpoint2 = get_two_check_points(args, ckpt_folder, exp_id_1=exp_id_1, exp_id_2=exp_id_2)

                    if args.train_or_test in ['test', 'test_on_noise']:
                        data_command += ' --only_eval'
                        result_folder_name, result_name = create_metric_folder(args, folder=ckpt_folder, name=f'curve{train_or_test_suffix}.npz')
                        log_name = os.path.join(result_folder_name, f'curve{train_or_test_suffix}')

                    command = f"python3 code/Mode_connectivity.py {data_command} " \
                            + f"--lr={args.curve_fitting_lr} " \
                            + f"--epochs={args.epochs} --to_eval={curve_folder}/checkpoint{args.result_suffix}-{args.eval_epoch}.pt " \
                            + f"--num_points={args.num_points} --save-frequency={args.save_frequency} --arch={args.arch} " \
                            + f"--dir={curve_folder} --curve=Bezier --num_bends={args.num_bends} --result-suffix={args.result_suffix} "\
                            + f"--init_start={checkpoint1} --init_end={checkpoint2} --fix_start --fix_end{command_suffix} " \
                            + f"--result-location {result_name} " \
                            + f"1>{log_name}.log 2>{log_name}.err"

                    BASH_COMMAND_LIST.append((command, result_name))
                
            elif args.experiment_type == 'train':
                
                if args.create_folder:
                    if not os.path.exists(ckpt_folder):
                        os.makedirs(ckpt_folder)
                
                stop_epoch = args.stop_epoch
                if args.efficient and compare_temperature_threshold(args, val):
                    stop_epoch = 50

                # This early stopping schedule follows from Charles and Michael's paper
                early_stopping_suffix = f' --save-early-stop --min-delta 0.0001 --patience 5'
                if args.training_type == 'normal':
                    command_suffix += f' --no-lr-decay --stop-epoch {stop_epoch}{early_stopping_suffix} --save-best'
                elif args.training_type == 'small_wd':
                    command_suffix += f' --no-lr-decay --stop-epoch {stop_epoch}{early_stopping_suffix} --save-best'
                elif args.training_type == 'lr_scaling':
                    command_suffix += f' --no-lr-decay --stop-epoch {stop_epoch}{early_stopping_suffix} --save-best'    
                    
                elif args.training_type == 'lr_decay':
                    # If we train with learning rate decay, we do not use the early stopping checkpoint
                    # This is because the early stopped checkpoint may confuse which temperature we use
                    command_suffix += f' --one-lr-decay --epochs 100 --save-best'
                    
                else:
                    raise NameError('Training type not included yet!')

                if args.ignore_incomplete_batch:
                    command_suffix += ' --ignore-incomplete-batch'
                    
                lr, bs, wd = get_training_params(args, val)
            
                exp_range = range(args.exp_start, args.exp_start+args.exp_num)
            
                for exp_id in exp_range:

                    result_name1 = os.path.join(ckpt_folder, f'net_exp_{exp_id}_early_stopped_model.pkl')
                    result_name2 = os.path.join(ckpt_folder, f'net_exp_{exp_id}.pkl')
                    result = (result_name1, result_name2)
            
                    command = f"python code/train.py --training-type {args.training_type} " \
                                + f"--arch {args.arch} --saving-folder {ckpt_folder} --file-prefix exp_{exp_id} " \
                                + f"--mixup-alpha 16.0 {data_command} " \
                                + f"--lr {lr} --weight-decay {wd} --train-bs {bs} " \
                                + f"{command_suffix} 1>{ckpt_folder}log_{exp_id}.txt 2>{ckpt_folder}err_{exp_id}.txt" 

                    BASH_COMMAND_LIST.append((command, result))
                        
            elif args.experiment_type == 'loss_acc':
                
                if args.early_stop_checkpoint:
                    command_suffix += ' --early-stopping'
                
                result_folder_name, result_name = create_metric_folder(args, folder=ckpt_folder, name=f'loss_acc{train_or_test_suffix}.pkl')

                command = f"python ./code/loss_acc.py " \
                          + f"--arch {args.arch} " \
                          + f"{data_command} --checkpoint-folder {ckpt_folder}{command_suffix} " \
                          + f"--result-location {result_name} " \
                          + f"1>{result_folder_name}/loss_acc{train_or_test_suffix}.log " \
                          + f"2>{result_folder_name}/loss_acc{train_or_test_suffix}.err"

                BASH_COMMAND_LIST.append((command, result_name))
                                
    return BASH_COMMAND_LIST


def clean_command_list(args, BASH_COMMAND_LIST):
    
    BASH_COMMAND_LIST_new = []
    
    for command, result in BASH_COMMAND_LIST:
        
        if args.check_result in ['first_result', 'second_result', 'single_result']:
            if args.check_result == 'first_result':
                single_result = result[0]
            elif args.check_result == 'second_result':
                single_result = result[1]
            elif args.check_result == 'single_result':
                single_result = result

            if not os.path.isfile(single_result):
                print(f"The result file {single_result} does not exist, and it is added to the list.")
                BASH_COMMAND_LIST_new.append(command)
            else:
                print(f"Result {single_result} already exists!")
                
        elif args.check_result == 'no_check':
            BASH_COMMAND_LIST_new.append(command)
            
        elif args.check_result == 'both_result':
            result1, result2 = result[0], result[1]
            # This means that we want both results to exist
            if (not os.path.isfile(result1)) or (not os.path.isfile(result2)):
                print(f"Result {result1} or {result2} does not exist, added to the list.")
                BASH_COMMAND_LIST_new.append(command)
            else:
                print(f"Both {result1} and {result2} already exist!")
        
        elif args.check_result == 'either_result':
            result1, result2 = result[0], result[1]
            # This means that we want either one of the two results to exist
            if (not os.path.isfile(result1)) and (not os.path.isfile(result2)):
                print(f"Result {result1} and {result2} both missing, added to the list.")
                BASH_COMMAND_LIST_new.append(command)
            else:
                print(f"Either {result1} or {result2} already exists!")
                
    return BASH_COMMAND_LIST_new


if __name__ == "__main__":

    print("Starting")    
    parser = argparse.ArgumentParser(description='Code for generating the bash scripts')
    parser.add_argument('--slurm-commands', default=False, action='store_true')
    parser.add_argument('--file-name', type=str, default = 'submit.sh', help='Name of the submission file')
    
    parser.add_argument('--create-folder', dest='create_folder', default = False, action='store_true',
                        help='should we create folder to store the checkpoint/results?')
    parser.add_argument('--check-result', type=str, default = 'single_result', 
                        choices = ['no_check', 'single_result', 'first_result', 'second_result', 'both_result', 'either_result'], 
                        help='should we check if the final result exists?')
    parser.add_argument('--experiment-type', type=str, default = 'train', 
                        choices = ['train', 'CKA', 'curve', 'dist', 'hessian', 'loss_acc'], 
                        help='which experiment do you want')

    # parameters for data
    parser.add_argument('--data-type', type=str, default='subset', 
                        choices = ['random_label', 'subset', 'subset_noisy', 'subset_augmentation', 'augmix'],
                        help='which type of data do you want to train with?')
    parser.add_argument('--train-or-test', type=str, default='train',
                        choices=['train', 'test'],
                        help='use training or testing to evaluate the metrics')
    parser.add_argument('--test-on-noise', dest='test_on_noise', default = False, action='store_true', 
                        help='change test data to have noisy labels also')
    parser.add_argument('--subset-noisy-prob', type=float, default=0.1, help='a fixed amount of subset noise')

    # parameters for training
    parser.add_argument('--arch', type=str, default = 'ResNet18', help='Model architecture')
    parser.add_argument('--training-type', type=str, default = 'normal', help='Training type', choices = ['normal', 'lr_decay', 'small_wd', 'lr_scaling'])
    parser.add_argument('--stop-epoch', type=int, default = 80, help='maximum number of epochs to train')
    parser.add_argument('--exp-num', type=int, default = 5, help='number of experiments')
    parser.add_argument('--exp-start', type=int, default = 0, help='start of the exp index')
    parser.add_argument('--different-width', dest='different_width', default = False, action='store_true',
                        help='Is the script for training with different width?')
    parser.add_argument('--width', type=int, default = 8, help='resnet18 width')
    
    parser.add_argument('--ignore-incomplete-batch', dest='ignore_incomplete_batch', default = False, action='store_true', help='ignore the last incomplete batch during training')
    parser.add_argument('--efficient', dest='efficient', default = False, action='store_true', help='train for 50 epochs for small batch size')
    parser.add_argument('--efficient_threshold', dest='efficient_threshold', type=float, default = 64, help='use efficient training when batch size <= 64')
    
    parser.add_argument('--lr-standard', type=float, default = 0.05, help='standard learning rate')
    parser.add_argument('--bs-standard', type=int, default = 128, help='standard batch size')
    parser.add_argument('--wd-standard', type=float, default = 5e-4, help='standard weight decay')
    
    # parameters for the 2D load-temperature phase plot
    parser.add_argument('--temperature-range', type=str, default = 'coarse', 
                        choices = ['coarse', 'fine', 'VGG_fine', 'customize', 'only_fine', 'DenseNet_fine', 'Efficient_fine'], 
                        help='use which set of temperature parameters')
    parser.add_argument('--temperature', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512, 1024],
                        help='which temperature parameters to use')
    parser.add_argument('--load-range', type=str, default = 'coarse', 
                        choices = ['subset_coarse', 'different_width_coarse', 'coarse', 'fine', 'only_fine', 'customize'], 
                        help='use which set of load parameters')
    parser.add_argument('--load', type=str, nargs='+', default=["0", "01", "02", "03", "04", "05", "06", "07", "08"],
                        help='which load parameters to use')
    parser.add_argument('--temperature-type', type=str, default='bs', help='which temperature parameter to measure')
    
    # parameters for curve fitting
    parser.add_argument('--save-frequency', type=int, default = 10, help='the frequency of saving a model')
    parser.add_argument('--curve-fitting-lr', type=float, default=0.01, metavar='LR', help='initial learning rate for curve fitting')
    parser.add_argument('--num-points', type=int, default = 61, help='number of points to save on each curve')
    parser.add_argument('--eval-epoch', type=int, default = 100, help='the epoch number of evaluate')
    parser.add_argument('--epochs', type=int, default = 100, help='training epochs')
    parser.add_argument('--early-stop-checkpoint', default = False, dest='early_stop_checkpoint', action='store_true',
                        help='use the early stopped checkpoint to measure')
    parser.add_argument('--result-suffix', type=str, default = '', help='curve result suffix')
    parser.add_argument('--multiple-curves', default = False, dest='multiple_curves', action='store_true',
                        help='average multiple curves')
    parser.add_argument('--num-bends', type=int, default = 3, help='number of bends to measure mode connectivity')
    
    # parameters for hessian
    parser.add_argument('--hessian-batch', type=int, default = 1, help='number of batches to measure hessian')
    parser.add_argument('--hessian-suffix', type=str, default = '', help='the suffix of hessian computation')
    parser.add_argument('--mini-hessian-batch-size', type=int, default = 200, help='size of each hessian batch')
    parser.add_argument('--name', type=str, default = 'cifar10_without_dataaugmentation', help='dataset name')
    
    # parameters for CKA
    parser.add_argument('--mixup-CKA', dest='mixup_CKA', default = False, action='store_true', 
                            help='measure CKA on mixup data')
    parser.add_argument('--mixup-alpha', type=float, default = 16, help='measure the CKA using mixup alpha =')
    parser.add_argument('--not-augment-CKA-for-data-augmentation-exp',  default = False, action='store_true')
    parser.add_argument('--CKA-batches',  type=int, default = 5)
    parser.add_argument('--CKA-repeat',  type=int, default = 1)

    

    args = parser.parse_args()
    
    if args.data_type == 'random_label':
        args.folder_suffix = 'random_label'
        args.CKA_suffix = 'random_label'
        args.dist_suffix = 'random_label'
        args.hessian_suffix = 'random_label'
    elif args.data_type == 'subset':
        args.folder_suffix = 'subset'
        args.CKA_suffix = 'subset'
        args.dist_suffix = 'subset'
        args.hessian_suffix = 'subset'
    elif args.data_type == 'subset_noisy':
        args.folder_suffix = 'subset_noisy'
        args.CKA_suffix = 'subset_noisy'
        args.dist_suffix = 'subset_noisy'
        args.hessian_suffix = 'subset_noisy'

    BASH_COMMAND_LIST = get_command_list(args)
    
    BASH_COMMAND_LIST = clean_command_list(args, BASH_COMMAND_LIST)
    
    script = get_script(args, BASH_COMMAND_LIST)
 
