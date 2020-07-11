from pathlib import Path

input_data_dir = Path('/kaggle/input/alaska2-image-steganalysis/')
output_data_dir = Path('/kaggle/working/data')
# input_data_dir = Path('/workdir/data/')
# output_data_dir = Path('/workdir/data/')

cover_dir = input_data_dir / 'Cover'
jmipod_dir = input_data_dir / 'JMiPOD'
juniward_dir = input_data_dir / 'JUNIWARD'
uerd_dir = input_data_dir / 'UERD'
sample_submission_path = input_data_dir / 'sample_submission.csv'
test_dir = input_data_dir / 'Test'

train_folds_path = output_data_dir / 'train_folds_v2.csv'
experiments_dir = output_data_dir / 'experiments'
predictions_dir = output_data_dir / 'predictions'
quality_json_path = output_data_dir / 'quality.json'
classes = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
unaltered_target = 0
class2target = {
    'Cover': unaltered_target,
    'JMiPOD': 1,
    'JUNIWARD': 2,
    'UERD': 3
}
qualities = [75, 90, 95]
quality2target = {q: t for t, q in enumerate(qualities)}
num_qualities = len(qualities)
class2altered = {cls: int(trg != unaltered_target)
                 for cls, trg in class2target.items()}
altered_classes = [cls for cls, alt in class2altered.items() if alt]
unaltered_classes = [cls for cls, alt in class2altered.items() if not alt]
altered_targets = [class2target[cls] for cls in altered_classes]
unaltered_targets = [class2target[cls] for cls in unaltered_classes]

num_unique_targets = len(set(class2target.values()))
n_folds = 2
folds = list(range(n_folds))
