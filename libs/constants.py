from pathlib import Path

data_path = Path('/home/lee/bat_research/data')

raw_path = data_path / 'raw'
processed_path = data_path / 'processed'
interim_path = data_path / 'interim'

raw_190526_path = raw_path / '190526'
pross_190526_path = processed_path / '190526'

raw_movie_190526_path = raw_190526_path / 'teshima movie' / '20190525'
raw_sound_190526_path = raw_190526_path / 'teshima sound' / '20190525'
