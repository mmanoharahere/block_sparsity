import os, time

def tsplit(string, delimiters):
    """Behaves str.split but supports multiple delimiters."""
    
    delimiters = tuple(delimiters)
    stack = [string,]
    
    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i+j, _substring)
            
    return stack


path_to_watch = "/data/ramyadML/scratch/block_sparsity/VGG16-ImageNet/"

mask_mix = "python /data/ramyadML/scratch/block_sparsity/checkpoint_analysis/save_masked_vgg.py --all_tensors "
eval_slim = "python /data/ramyadML/scratch/tensorflow-models/research/slim/eval_image_classifier.py --dataset_dir=/data/ramyadML/TF-slim-data/imageNet/processed/ --eval_dir=/tmp/tmp --model_name=vgg_16 --dataset_split_name=validation --labels_offset=1 "
#### UPDATE
checkpoint_dir="vgg16_apr23_monitor"

path_to_watch = path_to_watch + checkpoint_dir

checkpoint_analysis = "python /data/ramyadML/scratch/block_sparsity/checkpoint_analysis/weight_heatmap.py "

vgg_layers= ["vgg_16/conv1/conv1_1/",
        "vgg_16/conv1/conv1_2/",
        "vgg_16/conv2/conv2_1/",
        "vgg_16/conv2/conv2_2/",
        "vgg_16/conv3/conv3_1/",
        "vgg_16/conv3/conv3_2/",
        "vgg_16/conv3/conv3_3/",
        "vgg_16/conv4/conv4_1/",
        "vgg_16/conv4/conv4_2/",
        "vgg_16/conv4/conv4_3/",
        "vgg_16/conv5/conv5_1/",
        "vgg_16/conv5/conv5_2/",
        "vgg_16/conv5/conv5_3/",
        "vgg_16/fc6/",
        "vgg_16/fc7/",
        "vgg_16/fc8/",]

before = dict ([(f, None) for f in os.listdir (path_to_watch)])
visited_checkpoints = []

while 1:
    time.sleep (10)
    after = dict ([(f, None) for f in os.listdir (path_to_watch)])
    added = [f for f in after if not f in before]
    removed = [f for f in before if not f in after]
    # Added
    if added: 
        print "FILE Added: ", ", ".join(added)
        for newly_added in added:
            checkpoint = tsplit(newly_added,('.', '-'))
            if len(checkpoint) > 3:
                if checkpoint[2] not in visited_checkpoints:

                    #Checkpoint
                    visited_checkpoints.append(checkpoint[2])
                    print "CHECKPOINT:" + checkpoint[2]
                    filename="model.ckpt-"+checkpoint[2]
                    print "FILENAME:" + filename

                    #Mix Masks
                    command_mix = mask_mix + " --file_name=/data/ramyadML/scratch/block_sparsity/VGG16-ImageNet/" + checkpoint_dir + "/" + filename
                    print "COMMAND:" + command_mix
                    os.system(command_mix)

                    #Eval Part
                    command_eval = eval_slim + " --checkpoint_path=/tmp/model_ckpt_masked"
                    eval_log_filename = " 2>&1 | tee log_eval_" + checkpoint[2] + ".log"
                    print "COMMAND:" + command_eval + eval_log_filename
                    os.system(command_eval + eval_log_filename) 

                    #Checkpoint analysis part
                    for layer in vgg_layers:
                        command_analysis = checkpoint_analysis + " --file_name=/data/ramyadML/scratch/block_sparsity/VGG16-ImageNet/" + checkpoint_dir + "/" + filename  + " --tensor_name=" + layer + "weights" + " --mask=" + layer + "mask"
                        command_analysis_log = " 2>&1 | tee -a log_analysis_" + checkpoint[2] + ".log"
                        print "COMMAND:" + command_analysis + command_analysis_log
                        os.system(command_analysis + command_analysis_log)



    # Removed
    if removed: 
        print "FILE Removed: ", ", ".join (removed)

    before = after


