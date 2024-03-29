#%%
data_yaml = '''
path: datasets/  # dataset root dir
train: train_split.txt  # relative to path
val: val_split.txt  # relative to path
test: test_split.txt # relative to path

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
  10: C2
  11: C5
  12: E2
  13: E7
  14: EF2000
  15: F117
  16: F14
  17: F15
  18: F16
  19: F18
  20: F22
  21: F35
  22: F4
  23: J10
  24: J20
  25: JAS39
  26: KC135
  27: MQ9
  28: Mig31
  29: Mirage2000
  30: P3
  31: RQ4
  32: Rafale
  33: SR71
  34: Su25
  35: Su34
  36: Su57
  37: Tornado
  38: Tu160
  39: Tu95
  40: U2
  41: US2
  42: V22
  43: Vulcan
  44: XB70
  45: YF23
'''
# %%
with open('data.yaml', 'w') as f:
    f.write(data_yaml)
# %%
