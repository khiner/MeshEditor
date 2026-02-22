#include "SceneDefaults.h"

#include <glm/trigonometric.hpp>
SceneDefaults::SceneDefaults()
    : World{.Origin{0, 0, 0}, .Up{0, 1, 0}},
      ViewCamera{
          {0, 0, 2},
          {0, 0, 0},
          {Perspective{.FieldOfViewRad = glm::radians(60.f), .FarClip = 100.f, .NearClip = 0.01f}},
      },
      StudioLights{{
          {.Direction = {-0.854701f, 0.111111f, 0.507091f}, .Wrap = 0.200f, .DiffuseColor = {0.723042f, 0.723042f, 0.723042f}, .SpecularColor = {0.685956f, 0.685956f, 0.685956f}},
          {.Direction = {0.058607f, -0.987943f, -0.143295f}, .Wrap = 0.720f, .DiffuseColor = {0.063100f, 0.069978f, 0.067951f}, .SpecularColor = {0.145797f, 0.162642f, 0.157673f}},
          {.Direction = {0.972202f, 0.075846f, -0.221518f}, .Wrap = 0.281f, .DiffuseColor = {0.157432f, 0.163405f, 0.214035f}, .SpecularColor = {0.246195f, 0.225308f, 0.225308f}},
          {},
      }},
      AmbientColor{0, 0, 0},
      ViewportTheme{
          .Colors{
              .Wire{0, 0, 0},
              .WireEdit{0, 0, 0},
              .ObjectActive{1, 0.627f, 0.157f},
              .ObjectSelected{0.929f, 0.341f, 0},
              .Light{0, 0, 0, 0.314f},
              .Vertex{0, 0, 0},
              .VertexSelected{1, 0.478f, 0},
              .EdgeSelectedIncidental{1, 0.6f, 0},
              .EdgeSelected{1, 0.847f, 0},
              .FaceSelectedIncidental{1, 0.639f, 0, 0.2f},
              .FaceSelected{1, 0.718f, 0, 0.2f},
              .ElementActive{1, 1, 1, 0.2f},
              .FaceNormal{0.133f, 0.867f, 0.867f},
              .VertexNormal{0.137f, 0.380f, 0.867f},
              .Transform{1, 1, 1},
          },
          .SilhouetteEdgeWidth = 1,
      } {}
