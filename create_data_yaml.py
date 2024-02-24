#%%
data_yaml = '''
path: datasets/  # dataset root dir
train: train_split.txt  # relative to path
val: val_split.txt  # relative to path

# Classes
names:
  0: A10
  1: A400M
  2: AG600
  3: AV8B
  4: B1
  5: B2
  6: B52
  7: Be200
  8: C130
  9: C17
  10: C5
  11: E2
  12: EF2000
  13: F117
  14: F14
  15: F15
  16: F16
  17: F18
  18: F22
  19: F35
  20: F4
  21: J20
  22: JAS39
  23: MQ9
  24: Mig31
  25: Mirage2000
  26: RQ4
  27: Rafale
  28: SR71
  29: Su34
  30: Su57
  31: Tornado
  32: Tu160
  33: Tu95
  34: U2
  35: US2
  36: V22
  37: Vulcan
  38: XB70
  39: YF23
'''
# %%
with open('data.yaml', 'w') as f:
    f.write(data_yaml)
# %%
