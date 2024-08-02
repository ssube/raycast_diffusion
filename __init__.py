from .nodes.llm_nodes import GPT2Text
from .nodes.raycast_nodes import (
  EmptyMaterials,
  AddMaterial,
  CountMaterials,
  ExtrudeWorldGeometry,
  CameraControl,
  ProjectWorldLatents,
  MultiDiffControlInpaint,
  MultiControlImg2Img,
  UpdateWorldLatents,
  DebugMaterials,
  LoadCompilePipeline,
  LoadWorldLatents,
  MakeProfile,
  SaveWorldLatents,
  LoadModelGeometry,
)
from .nodes.utility_nodes import ExamineInput, Float3, ToString, ShowNumpyImage, SaveImageStack, ChangeDType, ChangeDevice, NormalizeArray

NODE_CLASS_MAPPINGS = {
  # llm
  "GPT2Text": GPT2Text,
  # raycast
  "EmptyMaterials": EmptyMaterials,
  "AddMaterial": AddMaterial,
  "CountMaterials": CountMaterials,
  "ExtrudeWorldGeometry": ExtrudeWorldGeometry,
  "CameraControl": CameraControl,
  "ProjectWorldLatents": ProjectWorldLatents,
  "MultiDiffControlInpaint": MultiDiffControlInpaint,
  "MultiControlImg2Img": MultiControlImg2Img,
  "UpdateWorldLatents": UpdateWorldLatents,
  "DebugMaterials": DebugMaterials,
  "LoadCompilePipeline": LoadCompilePipeline,
  "LoadWorldLatents": LoadWorldLatents,
  "MakeProfile": MakeProfile,
  "SaveWorldLatents": SaveWorldLatents,
  "LoadModelGeometry": LoadModelGeometry,
  # utility
  "Float3": Float3,
  "ToString": ToString,
  "ShowNumpyImage": ShowNumpyImage,
  "SaveImageStack": SaveImageStack,
  "ChangeDType": ChangeDType,
  "ChangeDevice": ChangeDevice,
  "NormalizeArray": NormalizeArray,
  "ExamineInput": ExamineInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  # llm
  "GPT2Text": "GPT2 Prompt Generator",
  # raycast
  "EmptyMaterials": "Create Empty Materials",
  "AddMaterial": "Add Material",
  "CountMaterials": "Count Materials",
  "DebugMaterials": "Debug Materials",
  "ExtrudeWorldGeometry": "Extrude World Geometry",
  "CameraControl": "Camera Control",
  "LoadCompilePipeline": "Load & Compile Pipeline",
  "MultiDiffControlInpaint": "Multi-Diff-Control Inpaint",
  "MultiControlImg2Img": "Multi-Control Img2Img",
  "LoadWorldLatents": "Load World Latents",
  "ProjectWorldLatents": "Project World Latents",
  "UpdateWorldLatents": "Update World Latents",
  "SaveWorldLatents": "Save World Latents",
  "MakeProfile": "Make Pipeline Profile",
  "LoadModelGeometry": "Load Model Geometry",
  # utility
  "ChangeDevice": "Change Device",
  "ChangeDType": "Change Data Type",
  "ExamineInput": "Examine Input",
  "Float3": "Float3 Vector",
  "NormalizeArray": "Normalize Numpy Array",
  "ShowNumpyImage": "Save Numpy Image",
  "SaveImageStack": "Save Image Stack",
  "ToString": "Convert To String",
}
