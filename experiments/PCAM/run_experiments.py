import argparse
import ast
import shlex
import os
import sys



from train import create_result_dir
from train import train_arg_parser



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_parameter_file(fname):
    # Read file line by line
    with open(fname) as f:
        content = f.readlines()
    # Return as list
    return [x.strip() for x in content]

def overwrite_parameter_file(fname, parameters):
    os.remove(fname) 
    with open(fname, "a") as f:
        for exp_str in parameters:
            f.write(exp_str)
            f.write('\n')



if __name__ == '__main__':

    ############################################## The run_experiments parameters parser
    parser_re = argparse.ArgumentParser()
    parser_re.add_argument('--continue_inprogress_first', type=str2bool, default=True)
    args_re = parser_re.parse_args()
    vargs_re = vars(args_re)
    continue_inprogress_first = vargs_re['continue_inprogress_first']

    ############################################## The parameter file names
    fname_todo       = './experiments/1_todo.txt'
    fname_inprogress = './experiments/2_inprogress.txt'
    fname_finished   = './experiments/3_finished.txt'

    ############################################# Check what experiments are schedules
    todo = read_parameter_file(fname_todo)
    inprogress = read_parameter_file(fname_inprogress)
    finished = read_parameter_file(fname_finished)

    while len(todo) + len(inprogress) > 0:
        ############################################# Check what experiments are schedules
        todo = read_parameter_file(fname_todo)
        inprogress = read_parameter_file(fname_inprogress)
        finished = read_parameter_file(fname_finished)
        # Print what we got:
        print('There {} experiments scheduled, {} in progress, {} already finished.'.format(len(todo),len(inprogress),len(finished)))

        ############################################## Select the experiment
        if continue_inprogress_first:
            if len(inprogress)>0:
                print('Continuing previously started experiment...')
                experiment = inprogress[0]
                del(inprogress[0])
                new_experimentQ = False
            elif len(todo)>0:
                print('Starting new experiment...')
                experiment = todo[0]
                del(todo[0])
                new_experimentQ = True
            else:
                print('No experiments to execute...')
                sys.exit()
        else:
            if len(todo)>0:
                print('Starting new experiment...')
                experiment = todo[0]
                del(todo[0])
                new_experimentQ = True
            elif len(inprogress)>0:
                print('Continuing previously started experiment...')
                experiment = inprogress[0]
                del(inprogress[0])
                new_experimentQ = False
            else:
                print('No experiments to execute...')
                sys.exit()

        ############################################### If a new experiment is started we create a new results directory
        # Get the default arguments and add the manually specified arguments (mainly to get to resultsdir_root)
        parser = train_arg_parser()
        args = parser.parse_args(shlex.split(experiment))
        vargs = vars(args)
        # Create results directory
        if new_experimentQ:
            resultdir = create_result_dir(vargs['resultdir_root'], vargs['modelfn'])
            experiment += ' --resultdir ' + resultdir

        ############################################## Update parameter files
        # Update in progress
        inprogress = inprogress + [experiment]
        overwrite_parameter_file(fname_inprogress, inprogress)
        # Remove from todo
        overwrite_parameter_file(fname_todo, todo)
        
        ################################################## Perform training
        print('-----------------------------------------------')
        print('-----------------------------------------------')
        os.system(sys.executable + ' train.py ' + experiment)
        print('-----------------------------------------------')
        print('-----------------------------------------------')

        ############################################## Update parameter files
        # Remove from in progress
        del(inprogress[-1])
        overwrite_parameter_file(fname_inprogress, inprogress)
        # And add to finished
        finished = finished + [experiment]
        overwrite_parameter_file(fname_finished, finished)
