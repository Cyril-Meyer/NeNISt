import os


def get(gpu_required=1, quit_if=False):
    name = os.uname()[1]
    # local machine, only 1 cuda gpu.
    if(name == 'lythandas'):
         os.environ["CUDA_VISIBLE_DEVICES"]='0'
    # gpu servers, 4 gpu per server.
    elif(name == 'lsiit-miv-gpu1' or name =='icube-images-gpu2'):
        import GPUtil
        # GPUtil.showUtilization()
        deviceIDs = GPUtil.getAvailable(order='load', limit=gpu_required, maxLoad=0.10, maxMemory=0.10)

        if len(deviceIDs) != gpu_required:
            print(os.linesep, "WARNING : REQUIREMENTS NOT MET", os.linesep)
            if quit_if:
                exit()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(deviceIDs).replace('[', '').replace(']', '').replace(' ', '')
    return os.environ["CUDA_VISIBLE_DEVICES"]