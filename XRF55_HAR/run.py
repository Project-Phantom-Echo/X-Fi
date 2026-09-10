import torch
from torch import nn
from torch.utils.data import DataLoader
from XRF55_Dataset import XRF55_Datase 
from utils import collate_fn_padd, har_train
from X_Fi import X_Fi
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser('X-Fi model for XRF55 HAR')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset, e.g. "D:/Data/XRF55/XRF_dataset')
    parser.add_argument('--backbone-root', type=str, default='./backbone_models')
    parser.add_argument(
        '--backbone-source',
        choices=('released', 'ours'),
        default='released',
        help='Verify either the released checkpoint hashes or our trainer metadata.',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('./pre-trained_weights'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available training resources:{device}')

    train_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=True)
    test_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=False)

    if len(train_dataset) != 15400 or len(test_dataset) != 6600:
        raise RuntimeError(
            f'Expected XRF55 Part-1 split 15400/6600, got {len(train_dataset)}/{len(test_dataset)}'
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_padd,
        num_workers=args.workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn_padd,
        num_workers=args.workers,
    )

    torch.manual_seed(args.seed)
    model = X_Fi(
        model_depth=5,
        num_classes=55,
        backbone_root=args.backbone_root,
        backbone_source=args.backbone_source,
    )
    model.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'configuration': (
            'released X-Fi XRF55 unified model'
            if args.backbone_source == 'released'
            else 'X-Fi XRF55 unified model with our trained backbones'
        ),
        'dataset': str(Path(args.dataset).resolve()),
        'backbone_root': str(Path(args.backbone_root).resolve()),
        'backbone_source': args.backbone_source,
        'backbone_paths': model.feature_extractor.backbone_paths,
        'backbone_sha256': model.feature_extractor.backbone_sha256,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'train_batch_size': args.train_batch_size,
        'test_batch_size': args.test_batch_size,
        'workers': args.workers,
        'seed': args.seed,
        'model_depth': 5,
        'modality_probabilities': {'mmwave': 0.5, 'wifi': 0.9, 'rfid': 0.6},
        'scheduler': None,
        'checkpoint_selection': 'fixed final epoch',
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))

    criterion = nn.CrossEntropyLoss()
    checkpoint = har_train(
        model = model,
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        num_epochs = args.epochs,
        learning_rate = args.learning_rate,
        criterion=criterion,
        device=device,
        save_dir = str(args.output_dir),
        val_random_seed = args.seed,
        checkpoint_name = 'checkpoint_final.pth',
            )
    print(f'Saved fixed-final checkpoint: {checkpoint}')

if __name__ == '__main__':
    main()
