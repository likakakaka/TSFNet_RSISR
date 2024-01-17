
from torch.utils.data import DataLoader

def create_dataloaders(args):
    """create dataloader"""
    if args.dataset == 'AID':
        from data.aid import AIDataset
        training_set = AIDataset(args, root_dir='.../AID-dataset/train',
                                 train=True)
        test_set = AIDataset(args, root_dir='.../AID-dataset/val',
                                   train=False)
    elif args.dataset == 'UCMerced':
        from data.ucmerced import UCMercedDataset
        training_set = UCMercedDataset(args, root_dir='.../UCMerced-dataset/train',
                                 train=True)
        test_set = UCMercedDataset(args, root_dir='.../UCMerced-dataset/val',
                                  train=False)
    elif args.dataset == 'NWPU':
        from data.nwpuresisc45 import NWPUDataset
        training_set = NWPUDataset(args, root_dir='.../NWPU-RESISC45/train',
                                 train=True)
        test_set = NWPUDataset(args, root_dir='.../NWPU-RESISC45/val',
                                  train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s ' % args.dataset)

    dataloaders = {'train': DataLoader(training_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0, drop_last=True),  # args.n_threads
                   'test': DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=0),
                   }  # args.n_threads

    return dataloaders



















