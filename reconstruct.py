import subprocess
import os
from tqdm import tqdm
import time

class goto1(Exception):
   pass
class goto2(Exception):
   pass
class goto3(Exception):
   pass


def create_forward_projection(forward_cmd):
    os.system(forward_cmd)
    for _ in tqdm(range(10), desc='Constructing forward projection'):
        time.sleep(0.5)

def reconstruct_image():
    reconstruction_cmd = 'python3 reconstruction.py'
    os.system(reconstruction_cmd)
    for _ in tqdm(range(10), desc='Reconstructing image'):
        time.sleep(0.5)
    os.chdir('../')

def main():
    forward_cmd = ''
    def loop():

            print('Please choose the forward projection image :')
            print('1. HotRod Phantom')
            print('2. Circle Phantom')
            print('3. Exit')
            reply = str(input())    
            try:
                if reply == '1':
                    raise goto1
                elif reply == '2':
                    raise goto2
                elif reply =='3':
                    print('Exiting ...')
                    return
                else:
                    raise goto3
            except goto1 as e:
                forward_cmd = 'python3 createphantom-hotrod.py'
                return forward_cmd
            except goto2 as e:
                forward_cmd = 'python3 createphantom-circle.py'
                return forward_cmd
            except goto3 as e:
                print('Invalid Entry. Please Retry.')
                return loop()
    forward_cmd = loop()
    if(forward_cmd == None):
        return

    print('Creating Forward Projection...')
    create_forward_projection(forward_cmd)
    print('Forward Projection Created')

    print('Reconstructing image ...')
    reconstruct_image()
    print('Image Reconstruction complete.')


if __name__ == '__main__':
    main()
