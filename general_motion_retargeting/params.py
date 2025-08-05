import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "booster_t1": ASSET_ROOT / "booster_t1" / "t1_mocap.xml",
    "booster_t1_4dof": ASSET_ROOT / "booster_t1" / "t1_mocap_4dof.xml",
    "stanford_toddy": ASSET_ROOT / "stanford_toddy" / "toddy_mocap.xml",
    "fourier_n1": ASSET_ROOT / "fourier_n1" / "n1_mocap.xml",
    "engineai_pm01": ASSET_ROOT / "engineai_pm01" / "pm_v2.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "booster_t1": IK_CONFIG_ROOT / "smplx_to_t1.json",
        "stanford_toddy": IK_CONFIG_ROOT / "smplx_to_toddy.json",
        "fourier_n1": IK_CONFIG_ROOT / "smplx_to_n1.json",
        "engineai_pm01": IK_CONFIG_ROOT / "smplx_to_pm01.json",
    },
    "bvh":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_to_g1.json",
        "booster_t1": IK_CONFIG_ROOT / "bvh_to_t1.json",
        "booster_t1_4dof": IK_CONFIG_ROOT / "bvh_to_t1_4dof.json",
        "fourier_n1": IK_CONFIG_ROOT / "bvh_to_n1.json",
        "stanford_toddy": IK_CONFIG_ROOT / "bvh_to_toddy.json",
        "engineai_pm01": IK_CONFIG_ROOT / "bvh_to_pm01.json",
    },
    "fbx":{
        "unitree_g1": IK_CONFIG_ROOT / "fbx_to_g1.json",
    },
}


ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "booster_t1": "Waist",
    "booster_t1_4dof": "Waist",
    "stanford_toddy": "waist_link",
    "fourier_n1": "base_link",
    "engineai_pm01": "LINK_BASE",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "booster_t1": 2.0,
    "booster_t1_4dof": 2.0,
    "stanford_toddy": 1.0,
    "fourier_n1": 2.0,
    "engineai_pm01": 2.0,
}