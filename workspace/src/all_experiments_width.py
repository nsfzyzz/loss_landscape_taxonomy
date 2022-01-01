import os

metrics = ["curve", "CKA", "hessian", "model_dist", "loss_acc"]

slurm_commands_customize = f'--nodes ace --num-gpus 4 --request-days 2 --log-folder'
tracking_commands_customize = f'--target-gpus 0 1 2 --memory-threshold 100 --gpu-query-time 1 --submit-wait-time 1'

###################################################################
# Would you like to customize the range of load and temperature?
###################################################################

customize_range_command = False
#customize_range_command = "--load-range customize --load 0 --temperature-range customize --temperature 128"


########################
# Train or test data?
########################

train_or_test_commands = " --train-or-test train"

#####################################
# Do we only generate slurm commands?
#####################################

#slurm_only_commands = ''
slurm_only_commands = ' --slurm-only-commands'

hessian_batch = 1

def customize_range(RANGE_COMMANDS):
    
    if customize_range_command:
        RANGE_COMMANDS = customize_range_command
    
    return RANGE_COMMANDS
        

def lab_commands(log_folder):
    
    # Common variables
    SLURM_COMMANDS = f"{slurm_commands_customize} {log_folder}"
    TRACKING_COMMANDS = f"--script-type tracking {tracking_commands_customize}"

    return SLURM_COMMANDS, TRACKING_COMMANDS


def different_metrics(COMMON_COMMANDS, FILE_NAME_SUFFIX,
                          SLURM_COMMANDS, TRACKING_COMMANDS, 
                          TRAINING_PARAMETERS, CURVE_PARAMETERS, CKA_PARAMETERS, LOSS_ACC_PARAMETERS,
                          DIST_PARAMETERS, HESSIAN_PARAMETERS, FILE_FOLDER='.'):
    
    # Training on the RISE machines
    if "training" in metrics:
        code = f"{COMMON_COMMANDS} {SLURM_COMMANDS} --file-name ./submissions/{FILE_FOLDER}/{FILE_NAME_SUFFIX}.sh {TRAINING_PARAMETERS} --check-result no_check --create-folder{slurm_only_commands}"
        os.system(code)
    
    # Mode connectivity code
    if "curve" in metrics:
        code = f"{COMMON_COMMANDS} {SLURM_COMMANDS} --file-name ./submissions/{FILE_FOLDER}/curve_fast_{FILE_NAME_SUFFIX}.sh {CURVE_PARAMETERS}{train_or_test_commands} --check-result no_check --create-folder{slurm_only_commands}"
        os.system(code)
    
    # CKA code
    if "CKA" in metrics:
        code = f"{COMMON_COMMANDS} {SLURM_COMMANDS} --file-name ./submissions/{FILE_FOLDER}/CKA_{FILE_NAME_SUFFIX}_mixup.sh {CKA_PARAMETERS}{train_or_test_commands} --check-result no_check --create-folder{slurm_only_commands}"
        os.system(code)
        
    # loss_acc code
    if "loss_acc" in metrics:
        code = f"{COMMON_COMMANDS} {SLURM_COMMANDS} --file-name ./submissions/{FILE_FOLDER}/loss_acc_{FILE_NAME_SUFFIX}.sh {LOSS_ACC_PARAMETERS}{train_or_test_commands} --check-result no_check --create-folder{slurm_only_commands}"
        os.system(code)

    # model dist code
    if "model_dist" in metrics:
        #code = f"{COMMON_COMMANDS} {TRACKING_COMMANDS} --command-file ./command_list_files/{FILE_FOLDER}/dist_{FILE_NAME_SUFFIX} {DIST_PARAMETERS} --check-result no_check{slurm_only_commands}"
        code = f"{COMMON_COMMANDS} {SLURM_COMMANDS} --file-name ./submissions/{FILE_FOLDER}/dist_{FILE_NAME_SUFFIX}.sh {DIST_PARAMETERS} --check-result no_check --create-folder{slurm_only_commands}"
        
        os.system(code)
    
    # hessian code
    if "hessian" in metrics:
        code = f"{COMMON_COMMANDS} {SLURM_COMMANDS} --file-name ./submissions/{FILE_FOLDER}/hessian_{FILE_NAME_SUFFIX}.sh {HESSIAN_PARAMETERS}{train_or_test_commands} --check-result single_result --create-folder{slurm_only_commands}"
        os.system(code)
    

def writing_experiments():
    
    # Specific variables for each experiment
    TRAINING_PARAMETERS="--stop-epoch 150 --training-type lr_decay --lr-standard 0.1 --ignore-incomplete-batch --exp-start 0 --exp-num 1"
    HESSIAN_PARAMETERS=f"--experiment-type hessian --hessian-batch {hessian_batch} --early-stop-checkpoint --training-type lr_decay"
    CKA_PARAMETERS="--experiment-type CKA --early-stop-checkpoint --mixup-CKA --mixup-alpha 16 --training-type lr_decay"
    DIST_PARAMETERS="--experiment-type model_dist --early-stop-checkpoint --training-type lr_decay"
    LOSS_ACC_PARAMETERS="--experiment-type loss_acc --early-stop-checkpoint --training-type lr_decay"

    ##########################################################
    # Training on different widths with the same subset config
    ##########################################################
    
    widths =["1", "2",  "3",  "4",  "6", "8",  "11",  "16", "23",  "32", "45",  "64", "91", "128"]
    
    subsets= ["10"] * len(widths)
    subset_nums= [1.0] * len(widths)

    for i in range(len(widths)):
        WIDTH=widths[i]
        SUBSET=subsets[i]
        ARCH=f"ResNet18_w{WIDTH}"
        EPOCH_NUM=int(10*1.0/subset_nums[i])*5
        print(f"For width={WIDTH} we use {EPOCH_NUM} epochs")
        
        # Here, we change the CURVE_PARAMETERS to different number of epochs, based on the size of data
        
        CURVE_PARAMETERS=f"--experiment-type curve --save-frequency=100 --num-points 5 --eval-epoch {EPOCH_NUM} --epochs {EPOCH_NUM} --early-stop-checkpoint --training-type lr_decay"
        
        # Here, we change the training epoch number to save time
        SLURM_COMMANDS, TRACKING_COMMANDS = lab_commands(log_folder = "width_lr_decay")

        ###################
        # Special attention
        ###################

        FILE_NAME_SUFFIX=f"width_{WIDTH}_lr_decay"
        WIDTH_COMMANDS=f"--different-width --width {WIDTH}"
        DATA_COMMANDS="--data-type subset"
        RANGE_COMMANDS=f"--load-range customize --load {SUBSET} --temperature-range Efficient_fine"
        RANGE_COMMANDS=customize_range(RANGE_COMMANDS)
        COMMON_COMMANDS=f"python write_script.py {WIDTH_COMMANDS} {DATA_COMMANDS} --arch {ARCH} {RANGE_COMMANDS}"

        different_metrics(COMMON_COMMANDS, FILE_NAME_SUFFIX,
                                SLURM_COMMANDS, TRACKING_COMMANDS, 
                                TRAINING_PARAMETERS, 
                                CURVE_PARAMETERS, 
                                CKA_PARAMETERS, 
                                LOSS_ACC_PARAMETERS, 
                                DIST_PARAMETERS,
                                HESSIAN_PARAMETERS, 
                                FILE_FOLDER="width_lr_decay")
        
writing_experiments()        
