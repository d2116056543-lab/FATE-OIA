$ErrorActionPreference = 'Stop'
Set-Location 'E:\sbw\SNNA_repro\SNNA'

$path = 'multi_label_train.py'
$text = Get-Content -LiteralPath $path -Raw
if ($text -match 'SNNA latest checkpoint with legacy 360 fallback') {
    Write-Host 'classifier_ckpt_patch_already_present'
    exit 0
}

$newResume = @"
    # Optionally resume from a checkpoint.
    # SNNA latest checkpoint with legacy 360 fallback: the original code saved
    # checkpoint.pth.tar but attempted to resume 360_checkpoint.pth.tar.
    to_restore = {"epoch": 0, "best_acc": 0.}
    resume_path = os.path.join(args.output_dir, "checkpoint.pth.tar")
    legacy_resume_path = os.path.join(args.output_dir, "360_checkpoint.pth.tar")
    if not os.path.isfile(resume_path) and os.path.isfile(legacy_resume_path):
        resume_path = legacy_resume_path
    utils.restart_from_checkpoint(
        resume_path,
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
"@

$resumePattern = '(?s)    # Optionally resume from a checkpoint\r?\n    to_restore = \{"epoch": 0, "best_acc": 0\.\}\r?\n    utils\.restart_from_checkpoint\(\r?\n        os\.path\.join\(args\.output_dir, "360_checkpoint\.pth\.tar"\),\r?\n        run_variables=to_restore,\r?\n        state_dict=linear_classifier,\r?\n        optimizer=optimizer,\r?\n        scheduler=scheduler,\r?\n    \)'
if ($text -notmatch $resumePattern) {
    throw 'Expected classifier resume block not found'
}
$text = [regex]::Replace($text, $resumePattern, $newResume, 1)

$oldVal = @"
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%")
            best_acc = max(best_acc, test_stats["acc"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
"@

$newVal = @"
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%")
            is_best = test_stats["acc"] >= best_acc
            best_acc = max(best_acc, test_stats["acc"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
"@

$valPattern = '(?s)            test_stats = validate_network\(val_loader, model, linear_classifier, args\.n_last_blocks, args\.avgpool_patchtokens\)\r?\n            print\(f"Accuracy at epoch \{epoch\} of the network on the \{len\(dataset_val\)\} test images: \{test_stats\[''acc''\]:\.1f\}%"\)\r?\n            best_acc = max\(best_acc, test_stats\["acc"\]\)\r?\n            print\(f''Max accuracy so far: \{best_acc:\.2f\}%''\)\r?\n            log_stats = \{\*\*\{k: v for k, v in log_stats\.items\(\)\},\r?\n                         \*\*\{f''test_\{k\}'': v for k, v in test_stats\.items\(\)\}\}'
if ($text -notmatch $valPattern) {
    throw 'Expected classifier validation block not found'
}
$text = [regex]::Replace($text, $valPattern, $newVal, 1)

$oldSave = @"
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
"@

$newSave = @"
            latest_path = os.path.join(args.output_dir, "checkpoint.pth.tar")
            torch.save(save_dict, latest_path)
            torch.save(save_dict, os.path.join(args.output_dir, "360_checkpoint.pth.tar"))
            if 'is_best' in locals() and is_best:
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best.pth.tar"))
"@

$savePattern = '            torch\.save\(save_dict, os\.path\.join\(args\.output_dir, "checkpoint\.pth\.tar"\)\)'
if ($text -notmatch $savePattern) {
    throw 'Expected classifier save block not found'
}
$text = [regex]::Replace($text, $savePattern, $newSave, 1)

Set-Content -LiteralPath $path -Value $text -Encoding UTF8
Write-Host 'classifier_ckpt_patch_updated'
