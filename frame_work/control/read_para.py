import yaml

with open("functional grasp/frame_work/config/para.yaml") as ymlfile:
    cfg = yaml.load_all(ymlfile, Loader=yaml.SafeLoader)

    for data in cfg:
        print(data['glass_wine']['handoff']["index"])