import os
import argparse


parser = argparse.ArgumentParser(description="AML Generic Launcher")
parser.add_argument("--workdir", default="", 
                        help="The working directory.")
parser.add_argument("--cfg", default="")
args, _ = parser.parse_known_args()

# start training
os.chdir(args.workdir)
# The train.py is in the blob under the args.workdir
# os.system("WORKDIR=%s && python train.py --model_cfg %s --solver_cfg %s --evaluator_cfg %s --cuda" % (args.workdir, model_cfg, solver_cfg, evaluator_cfg))

'''
Change the command you want to execute here
'''
filename = args.cfg
if filename.endswith('.sh'):
    filename = filename[:-3]
os.system("cd ./projects/SSSeg/PixelSSL/ && python setup.py install && cd ./task/sseg && bash aml_script/{}.sh {}".format(filename, args.workdir))

# os.system("ls && pwd")
