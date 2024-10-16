import logging
import os
import random
import sys
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import DiceLoss
from .load_data import data_ld


def trainer_ACDC(args, model, snapshot_path):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, force=True,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    db_train, db_validation = data_ld(args.processed_root, ref=args.img_size, batch=args.batch_size)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_validation, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                           worker_init_fn=worker_init_fn)

    torch.autograd.set_detect_anomaly(True)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 1
    tic = timeit.default_timer()
    for epoch_num in iterator:
        loss_tot = 0
        val_loss_tot=0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            iter_num = iter_num + 1
            loss_tot = loss_tot + loss.item()

            loss_tot_mean = loss_tot / len(trainloader)

            if iter_num % 10 == 0:
                logging.info('epoch: %d  iteration: %d  loss: %f' % (epoch_num, iter_num, loss.item()))

        writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch_num)
        writer.add_scalars('info/total_loss', {'train': loss_tot_mean}, epoch_num)


        for i_batch, sampled_batch in enumerate(valloader):
            model.eval()
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            val_loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
            val_loss = 0.4 * val_loss_ce + 0.6 * loss_dice
            val_loss_tot = val_loss_tot + val_loss.item()


        val_loss_tot_mean = val_loss_tot/len(valloader)
        writer.add_scalars('info/total_loss', {'val':val_loss_tot_mean}, epoch_num)
        logging.info(' val_loss : %f' % (val_loss_tot_mean))

        scheduler.step(val_loss_tot_mean)



        if val_loss_tot_mean < best_loss:
            best_loss = val_loss_tot_mean

            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num % 50 == 0:
            if epoch_num != 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save({
                    'epoch': epoch_num ,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if epoch_num % 5 == 0:
            image = image_batch[0, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image, iter_num)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
            labs = label_batch[0, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, iter_num)


        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    toc = timeit.default_timer()
    benefit_time = toc - tic
    print('benefit time:', benefit_time)
    return "Training Finished!"