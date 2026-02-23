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
          {.Direction = {0.000000f, 0.639175f, 0.769061f}, .Wrap = 0.100000f, .DiffuseColor = {0.507074f, 0.507074f, 0.507074f}, .SpecularColor = {0.666141f, 0.666141f, 0.666141f}},
          {.Direction = {-0.846939f, -0.357143f, 0.393883f}, .Wrap = 0.340000f, .DiffuseColor = {0.021936f, 0.058160f, 0.063719f}, .SpecularColor = {0.033668f, 0.060061f, 0.063712f}},
          {.Direction = {0.755102f, -0.530612f, 0.385060f}, .Wrap = 0.340000f, .DiffuseColor = {0.063721f, 0.040061f, 0.037017f}, .SpecularColor = {0.069396f, 0.046455f, 0.046455f}},
          {.Direction = {0.034483f, 0.913793f, 0.404714f}, .Wrap = 0.724138f, .DiffuseColor = {0.000000f, 0.000000f, 0.000000f}, .SpecularColor = {0.023079f, 0.023393f, 0.025394f}},
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
