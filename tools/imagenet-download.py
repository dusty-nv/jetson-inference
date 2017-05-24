#!/usr/bin/env python
# Copyright (c) 2014 Seiya Tokui
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import imghdr
import Queue
import os
import socket
import sys
import tempfile
import threading
import time
import urllib2
import glob

def download(url, timeout, retry, sleep, verbose=False):
    """Downloads a file at given URL."""
    count = 0
    while True:
        try:
            f = urllib2.urlopen(url, timeout=timeout)
            if f is None:
                raise Exception('Cannot open URL {0}'.format(url))
            content = f.read()
            f.close()
            break
        except urllib2.HTTPError as e:
            if 500 <= e.code < 600:
                if verbose:
                    sys.stderr.write('Error: HTTP with code {0}\n'.format(e.code))
                count += 1
                if count > retry:
                    if verbose:
                        sys.stderr.write('Error: too many retries on {0}\n'.format(url))
                    raise
            else:
                if verbose:
                    sys.stderr.write('Error: HTTP with code {0}\n'.format(e.code))
                raise
        except urllib2.URLError as e:
            if isinstance(e.reason, socket.gaierror):
                count += 1
                time.sleep(sleep)
                if count > retry:
                    if verbose:
                        sys.stderr.write('Error: too many retries on {0}\n'.format(url))
                    raise
            else:
                if verbose:
                    sys.stderr.write('Error: URLError {0}\n'.format(e))
                raise
        #except Exception as e:
        #    if verbose:
        #        sys.stderr.write('Error: unknown during download: {0}\n'.format(e))
    return content

def imgtype2ext(typ):
    """Converts an image type given by imghdr.what() to a file extension."""
    if typ == 'jpeg':
        return 'jpg'
    if typ is None:
        raise Exception('Cannot detect image type')
    return typ

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def download_imagenet(list_filename,
                      out_dir,
                      timeout=10,
                      retry=10,
                      num_jobs=1,
                      sleep_after_dl=1,
                      verbose=False,
                      offset=0,
                      msg=1):
    """Downloads to out_dir all images whose names and URLs are written in file
    of name list_filename.
    """

    make_directory(out_dir)

    count_total = 0
    with open(list_filename) as list_in:
        for i, l in enumerate(list_in):
            pass
        count_total = i + 1
    count_total -= offset

    sys.stderr.write('Total: {0}\n'.format(count_total))
    
    num_jobs = max(num_jobs, 1)

    entries = Queue.Queue(num_jobs)
    done = [False]

    counts_fail = [0 for i in xrange(num_jobs)]
    counts_success = [0 for i in xrange(num_jobs)]

    def producer():
        count = 0
        with open(list_filename) as list_in:
            for line in list_in:
                if count >= offset:
                    name, url = line.strip().split(None, 1)
                    entries.put((name, url), block=True)
                count += 1

        entries.join()
        done[0] = True

    def consumer(i):
        while not done[0]:
            try:
                name, url = entries.get(timeout=1)
            except:
                continue

            try:
                if name is None:
                    if verbose:
                        sys.stderr.write('Error: Invalid line: {0}\n'.line)
                    counts_fail[i] += 1
                    continue

                directory = os.path.join(out_dir, name.split('_')[0])
                rpath = os.path.join(directory, '{0}.*'.format(name))
                lf = glob.glob(rpath)
                if lf:
                    print "skipping: already have", lf[0]
                    counts_success[i] += 1
                    entries.task_done()
                    continue

                content = download(url, timeout, retry, sleep_after_dl)
                ext = imgtype2ext(imghdr.what('', content))
                try:
                    make_directory(directory)
                except:
                    pass
                path = os.path.join(directory, '{0}.{1}'.format(name, ext))
                with open(path, 'w') as f:
                    f.write(content)
                counts_success[i] += 1
                time.sleep(sleep_after_dl)

            except Exception as e:
                counts_fail[i] += 1
                if verbose:
                    sys.stderr.write('Error: {0} / {1}: {2}\n'.format(name, url, e))
            
            entries.task_done()

    def message_loop():
        if verbose:
            delim = '\n'
        else:
            delim = '\r'

        while not done[0]:
            count_success = sum(counts_success)
            count = count_success + sum(counts_fail)
            rate_done = count * 100.0 / count_total
            if count == 0:
                rate_success = 0
            else:
                rate_success = count_success * 100.0 / count
            sys.stderr.write(
                '{0} / {1} ({2}%) done, {3} / {0} ({4}%) succeeded                    {5}'.format(
                    count, count_total, rate_done, count_success, rate_success, delim))

            time.sleep(msg)
        sys.stderr.write('done')

    producer_thread = threading.Thread(target=producer)
    consumer_threads = [threading.Thread(target=consumer, args=(i,)) for i in xrange(num_jobs)]
    message_thread = threading.Thread(target=message_loop)

    producer_thread.start()
    for t in consumer_threads:
        t.start()
    message_thread.start()

    # Explicitly wait to accept SIGINT
    try:
        while producer_thread.isAlive():
            time.sleep(1)
    except:
        sys.exit(1)

    producer_thread.join()
    for t in consumer_threads:
        t.join()
    message_thread.join()

    sys.stderr.write('\ndone\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('list', help='Imagenet list file')
    p.add_argument('outdir', help='Output directory')
    p.add_argument('--jobs', '-j', type=int, default=1,
                   help='Number of parallel threads to download')
    p.add_argument('--timeout', '-t', type=int, default=10,
                   help='Timeout per image in seconds')
    p.add_argument('--retry', '-r', type=int, default=10,
                   help='Max count of retry for each image')
    p.add_argument('--sleep', '-s', type=float, default=1,
                   help='Sleep after download each image in second')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Enable verbose messages')
    p.add_argument('--offset', '-o', type=int, default=0,
                   help='Offset to where to start in Imagenet list file')
    p.add_argument('--msg', '-m', type=int, default=1,
                   help='Logging message every x seconds')
    args = p.parse_args()

    download_imagenet(args.list, args.outdir,
                      timeout=args.timeout, retry=args.retry,
                      num_jobs=args.jobs, verbose=args.verbose, 
                      offset=args.offset, msg=args.msg)
