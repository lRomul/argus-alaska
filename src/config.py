from pathlib import Path


work_dir = Path('/workdir')
data_dir = work_dir / 'data'
cover_dir = data_dir / 'Cover'
jmipod_dir = data_dir / 'JMiPOD'
juniward_dir = data_dir / 'JUNIWARD'
uerd_dir = data_dir / 'UERD'
sample_submission_path = data_dir / 'sample_submission.csv'
test_dir = data_dir / 'Test'

train_folds_path = data_dir / 'train_folds.csv'
classes = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
class2target = {
    'Cover': 0,
    'JMiPOD': 1,
    'JUNIWARD': 1,
    'UERD': 1
}
n_folds = 5
