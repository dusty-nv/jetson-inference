#!/usr/bin/env python3
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import os
import argparse
import subprocess
import pprint

from jetson_utils import videoSource

parser = argparse.ArgumentParser()

parser.add_argument('--module', type=str, default=None, help='limit testing to a specific module (i.e. "detectnet"')
parser.add_argument('--threshold', type=float, default=0.002, help='threshold for percentage difference between pixels')
parser.add_argument('--generate', action='store_true', help='generate the expected outputs')
parser.add_argument('--no-python', action='store_true', help='skip testing of the python modules')
parser.add_argument('--python-only', action='store_true', help='only test the python modules')
parser.add_argument('--stop-on-failure', action='store_true', help='stop testing as soon as any testing failure is encountered')
parser.add_argument('--verbose', action='store_true', help='view the subprocess output')

args = parser.parse_args()
print(args)

imagenet_images = [
    'cat_*.jpg',
    'granny_smith_*.jpg',
    'black_bear.jpg',
    'brown_bear.jpg',
    'polar_bear.jpg',
    'jellyfish.jpg',
]

tests = {

    'imagenet' : {
        'googlenet' : imagenet_images,
        'resnet18' : imagenet_images
    },
            
    'detectnet' : {
        'ssd-mobilenet-v2' : [
            'peds_*.jpg',
            'humans_*.jpg',
            'object_*.jpg'
        ]
    },

    'segnet' : {
        'fcn-resnet18-cityscapes-512x256' : ['city_*.jpg'],
        'fcn-resnet18-deepscene-576x320'  : ['trail_*.jpg'],
        'fcn-resnet18-mhp-512x320'        : ['humans_*.jpg'],
        'fcn-resnet18-pascal-voc-320x320' : ['object_*.jpg'],
        'fcn-resnet18-sun-rgbd-512x400'   : ['room_*.jpg'],
    },
    
    'posenet' : {
        'resnet18-body' : ['humans_*.jpg']
    },
    
    'depthnet' : {
        'monodepth-fcn-mobilenet' : ['room_*.jpg']
    },
}

pprint.pprint(tests)

results = {}
event_log = []

def log(str):
    print(str)
    event_log.append(str)

def image_diff(img_a, img_b):   
    shape = img_a.shape
    value = 0.0
    
    for y in range(shape[0]):
        for x in range(shape[1]):
            px_a = img_a[y,x]
            px_b = img_b[y,x]
            
            for c in range(shape[2]):
                value += abs(px_a[c] - px_b[c]) #abs(img_a[y,x,c] - img_b[y,x,c])

    return value / float(shape[0] * shape[1] * shape[2] * 255)
    
def compare_images(path_a, path_b):
    source_a = videoSource(path_a)
    source_b = videoSource(path_b)

    idx = 0
    
    while True:
        try:
            img_a = source_a.Capture()
            img_b = source_b.Capture()
        except:
            if idx > 0:
                return True
            else:
                log(f'[FAIL]   failed to load any images from sequence "{os.path.basename(path_a)}"')
                return False
                
        if img_a.shape != img_b.shape:
            log(f'[FAIL]   image {idx} from sequence "{os.path.basename(path_a)}" had different dimensions:')
            log(f'   {path_a}  {img_a.shape}')
            log(f'   {path_b}  {img_b.shape}')
            return False
        
        diff = image_diff(img_a, img_b)
        
        if diff > 0:
            log(f'[{"FAIL" if diff > args.threshold else "WARN"}]   image {idx} from sequence "{os.path.basename(path_a)}" had pixel difference of {diff} ({"exceeding" if diff > args.threshold else "within"} threshold of {args.threshold})')
            
            if diff > args.threshold:
                return False
            
        idx += 1
        
    return True
    
def test_images(module, model, images):
    print(f'testing {module}, model {model}, images {images}')
    
    output_dir_gt = 'images/qa/groundtruth'
    output_dir_results = 'images/qa/results'
        
    # construct input/output paths
    inputs = f'images/{images}'
    outputs_gt = f'{output_dir_gt}/{module}_{model}_{images}'
    outputs_results = f'{output_dir_results}/{module}_{model}_{images}'
   
    if args.generate:
        outputs = outputs_gt
    else:
        outputs = outputs_results
        
    # run command
    cmd = f'{module} --network={model} "{inputs}" "{outputs}"'
    print(cmd)
    cmd_results = subprocess.run(cmd, shell=True, stdout=None if args.verbose else subprocess.DEVNULL)
    
    if cmd_results.returncode != 0:
        log(f'[FAIL]   return code {cmd_results.returncode}:  {cmd}')
        return False
        
    if args.generate:
        return True
    
    return compare_images(outputs_gt, outputs_results)
    
def test_model(module, model, images):
    print(f'testing {module}, model {model}')
    
    for image in images:
        result = test_images(module, model, image)
    
        if module not in results:
            results[module] = {}
            
        if model not in results[module]:
            results[module][model] = {}
            
        results[module][model][image] = 'PASSED' if result else 'FAILED'
        
        if not result and args.stop_on_failure:
            raise Exception('encountered an error and --stop-on-failure was specified')
        
        
def test_module(module, models):
    print(f'testing {module}')
    
    for model in models.keys():
        test_model(module, model, models[model])
    
def run_tests():
    for module in tests.keys():
        if args.module is not None and module.lower() != args.module.lower():
            continue
            
        models = tests[module]
        
        if not args.python_only:
            test_module(module, models)
            
        if not args.no_python:
            test_module(module + '.py', models)
        
    print('\n')
    print('EVENT LOG:\n')
    
    for str in event_log:
        print(str)

    print('\n')
    print('RESULTS SUMMARY:\n')
    pprint.pprint(results)
    
if __name__ == '__main__':
    run_tests()