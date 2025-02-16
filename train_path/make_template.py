import copy
import os
import argparse

import ruamel.yaml
import yaml

# 读取配置文件模板
with open('/home/ymt/yolov5/config_data.yaml') as f:
    config = yaml.safe_load(f)
    model_name = config['temp_yaml']['model_name']
    hz_classes = config['temp_yaml']['hz_classes']
    hz_classes_ok_num = config['temp_yaml']['hz_classes_ok_num']
    is_four_caid = config['temp_yaml']['is_four_caid']
    four_class = config['temp_yaml']['four_class']
    center_class = config['temp_yaml']['center_class']
    class_ok = config['temp_yaml']['class_ok']
    class_ng = config['temp_yaml']['class_ng']
    left = config['temp_yaml']['left']
    wz_class_name = config['temp_yaml']['wz_class_name']
    wz_class = config['temp_yaml']['wz_class']
    x_check_ok = config['temp_yaml']['x_check_ok']
    y_check_ok = config['temp_yaml']['y_check_ok']
    new_model=config['temp_yaml']['new_model']

if is_four_caid:
    # 读取配置文件模板
    with open('template_01.yaml') as f:
        template = yaml.safe_load(f)
        template['weights_path'] = f"components/biz/weights/{model_name}.pt"
        template['hz_classes'] = hz_classes
        template['hz_classes_ok_num'] = hz_classes_ok_num
        template['center_camera']['hz_classes'] = center_class
        template['hz_check_name_ok'] = class_ok
        template['hz_check_name_ng'] = class_ng
        if left:
            template['dingtiao']['dt_ca_names'] = [5, 7]
        else:
            template['dingtiao']['dt_ca_names'] = [6]
        template['wz']['wz_class_name'] = wz_class_name
        template['wz']['wz_class'] = wz_class
        template['wz']['x_check_ok'] = x_check_ok
        template['wz']['y_check_ok'] = y_check_ok
        template['new_model'] = new_model

else:
    with open('template_02.yaml') as f:
        template = yaml.safe_load(f)
        template['weights_path'] = f"components/biz/weights/{model_name}.pt"
        template['hz_classes'] = hz_classes
        template['hz_classes_ok_num'] = hz_classes_ok_num
        template['center_camera']['hz_classes'] = center_class
        template['hz_check_name_ok'] = class_ok
        template['hz_check_name_ng'] = class_ng
        if left:
            template['dingtiao']['dt_ca_names'] = [1, 3]
        else:
            template['dingtiao']['dt_ca_names'] = [2]
        template['wz']['wz_class_name'] = wz_class_name
        template['wz']['wz_class'] = wz_class
        template['wz']['x_check_ok'] = x_check_ok
        template['wz']['y_check_ok'] = y_check_ok
        template['new_model'] = new_model

output_path = f"config/{model_name}.yaml"

# 写入新的配置文件
wryaml = ruamel.yaml.YAML()
wryaml.default_flow_style = False
with open(output_path, 'w') as f:
    # yaml.safe_dump(template, f,sort_keys=False)
    wryaml.dump(template, f)

print(f"创建模版{model_name}.yaml完成！")
