from option import args
import model
import torch
import cv2
import glob
import utils
import os
import numpy as np
import math
import time
def calculate_psnr(img1, img2):
    """calculate psnr
    img1: (B) x H x W x C
    img2: (B) x H x W x C
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255. / math.sqrt(mse))


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
if __name__ == '__main__':
    device = torch.device('cpu' if args.cpu else 'cuda')
    args.model='tsfnet'
    root_dir = '.../AID-dataset/test'
    # root_dir = '.../NWPU-RESISC45/test'
    # root_dir = '.../UCMerced-dataset/test'
    args.pre_train = '.../tsf_aidx16.pt'
    args.dir_out= '.../aidx16/tsf_aidx16/'
    args.rgb_range = 1.
    args.cubic_input=False
    save_img = True
    args.scale = [16]
    args.patch_size = 256
    args.test_block = False
    dir_hr = os.path.join(root_dir, 'HR_x' + str(args.scale[0]))
    dir_lr = os.path.join(root_dir, 'LR_x' + str(args.scale[0]))
    list_hr = glob.glob(os.path.join(dir_hr, '*.png'))
    # list_hr = glob.glob(os.path.join(dir_hr, '*.jpg'))
    # list_hr = glob.glob(os.path.join(dir_hr, '*.tif'))
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)
    img_dir = os.path.join(args.dir_out, 'net_img_result')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    open_type = 'w'
    log_file = open(args.dir_out + '/log_metric.txt', open_type)
    args.resume = 0
    args.print_model = False
    args.test_only = True
    checkpoint = utils.checkpoint(args)
    sr_model = model.Model(args, checkpoint, device='cuda')

    sr_model.eval()

    list_lr = []
    crop_border = args.scale[0]
    eval_acc = 0
    eval_pnsr_acc = 0
    eval_ssim_acc = 0
    eval_sam_acc=0
    eval_qi_acc=0
    eval_scc_acc=0
    img_num = 0
    start = time.time()
    with torch.no_grad():
        for i in range(len(list_hr)):
            filename = os.path.split(list_hr[i])[-1]
            lr_path = os.path.join(dir_lr, filename)
            print(lr_path)
            if args.cubic_input:
                hr = cv2.imread(list_hr[i], cv2.IMREAD_COLOR)
                lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
                lr = cv2.resize(lr, (hr.shape[0], hr.shape[1]), interpolation=cv2.INTER_CUBIC)
                hr_np = hr.astype(np.float32) / 255. *args.rgb_range
                lr_np = lr.astype(np.float32) / 255. *args.rgb_range
            else:
                hr_np = cv2.imread(list_hr[i], cv2.IMREAD_COLOR).astype(np.float32) / 255. *args.rgb_range
                lr_np = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255. *args.rgb_range

            lr = np.transpose(lr_np if lr_np.shape[2] == 1 else lr_np[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
            lr = torch.from_numpy(lr).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

            if args.test_block:
                # test block-by-block
                b, c, h, w = lr.shape
                factor = args.scale[0] if not args.cubic_input else 1
                tp = args.patch_size
                if not args.cubic_input:
                    ip = tp // factor
                else:
                    ip = tp

                assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
                if not args.cubic_input:
                    sr = torch.zeros((b, c, h * factor, w * factor))
                else:
                    sr = torch.zeros((b, c, h, w))

                for iy in range(0, h, ip-2):

                    if iy + ip > h:
                        iy = h - ip
                    ty = factor * iy

                    for ix in range(0, w, ip-2):

                        if ix + ip > w:
                            ix = w - ip
                        tx = factor * ix

                        # forward-pass
                        lr_p = lr[:, :, iy:iy + ip, ix:ix + ip]
                        sr_p,_= sr_model(lr_p)
                        if iy==0:
                            iy_min=0
                        else:
                            iy_min=iy+1
                        if ix == 0:
                            ix_min = 0
                        else:
                            ix_min = ix+1
                        if iy + ip==h:
                            iy_max=iy + ip
                        else:
                            iy_max=iy + ip-1
                        if ix + ip==w:
                            ix_max = ix + ip
                        else:
                            ix_max = ix + ip-1
                        sr[:, :,iy_min*factor:iy_max*factor, ix_min*factor:ix_max*factor]=sr_p[:,:,(iy_min-iy)*factor:(iy_max-iy)*factor,(ix_min-ix)*factor:(ix_max-ix)*factor]
            else:
                sr, _ = sr_model(lr)
            output = sr.data.squeeze().float().cpu().clamp_(0, 1 * args.rgb_range).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0 / args.rgb_range).round().astype(np.uint8)  # float32 to uint8
            img_gt = hr_np * 255.0 / args.rgb_range  # float32 to uint8
            img_gt = np.squeeze(img_gt)
            ####保存图像
            if save_img:
                cv2.imwrite(os.path.join(img_dir, filename), output)
                print('saving img!')
            # crop borders
            if crop_border == 0:
                cropped_hr = img_gt
                cropped_sr = output
            else:
                cropped_hr = img_gt[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_sr = output[crop_border:-crop_border, crop_border:-crop_border, :]
            sig_pnsr = calculate_psnr(cropped_sr, cropped_hr)
            eval_pnsr_acc += sig_pnsr
            sig_ssim = calculate_ssim(cropped_sr, cropped_hr)
            eval_ssim_acc += sig_ssim

            img_num += sr.shape[0]

            ssim_acc = eval_ssim_acc / img_num
            pnsr_acc = eval_pnsr_acc / img_num
            log = '[{} {}x{}] {:3d} {}\t{}:{:.6f} {}:{:.6f}'.format(
                args.model,
                args.dataset,
                crop_border,
                i + 1,
                filename,
                'ssim',
                sig_ssim,
                'pnsr',
                sig_pnsr
            )
            log_file.write(log + '\n')
            print(log)
        log = 'Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
            pnsr_acc,
            ssim_acc)
        log_file.write(log + '\n')
        print(log)
        log_file.close()
        end = time.time()
        print("测试运行时间:%.2f秒" % (end - start))
