// Copyright 2023-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
//
// Minimal local copy of the preview XR_MNDX_xdev_space declarations needed by
// the camera overlay. Keeping this header local makes the migrated
// camera_streamer build context self-contained.

#pragma once

#include <openxr/openxr.h>

#define XR_ENUM(type, enm, constant) static const type enm = (type)constant
#define XR_STRUCT_ENUM(enm, constant) XR_ENUM(XrStructureType, enm, constant)

#define XR_MNDX_xdev_space 1
#define XR_MNDX_xdev_space_SPEC_VERSION 2
#define XR_MNDX_XDEV_SPACE_EXTENSION_NAME "XR_MNDX_xdev_space"

XR_DEFINE_ATOM(XrXDevIdMNDX)
XR_DEFINE_HANDLE(XrXDevListMNDX)

XR_STRUCT_ENUM(XR_TYPE_CREATE_XDEV_LIST_INFO_MNDX, 1000444002);
typedef struct XrCreateXDevListInfoMNDX
{
    XrStructureType type;
    const void* XR_MAY_ALIAS next;
} XrCreateXDevListInfoMNDX;

XR_STRUCT_ENUM(XR_TYPE_GET_XDEV_INFO_MNDX, 1000444003);
typedef struct XrGetXDevInfoMNDX
{
    XrStructureType type;
    const void* XR_MAY_ALIAS next;
    XrXDevIdMNDX id;
} XrGetXDevInfoMNDX;

XR_STRUCT_ENUM(XR_TYPE_XDEV_PROPERTIES_MNDX, 1000444004);
typedef struct XrXDevPropertiesMNDX
{
    XrStructureType type;
    void* XR_MAY_ALIAS next;
    char name[256];
    char serial[256];
    XrBool32 canCreateSpace;
} XrXDevPropertiesMNDX;

XR_STRUCT_ENUM(XR_TYPE_CREATE_HAND_TRACKER_XDEV_MNDX, 1000444006);
typedef struct XrCreateHandTrackerXDevMNDX
{
    XrStructureType type;
    const void* XR_MAY_ALIAS next;
    XrXDevListMNDX xdevList;
    XrXDevIdMNDX id;
} XrCreateHandTrackerXDevMNDX;

typedef XrResult(XRAPI_PTR* PFN_xrCreateXDevListMNDX)(XrSession session,
                                                      const XrCreateXDevListInfoMNDX* info,
                                                      XrXDevListMNDX* xdevList);
typedef XrResult(XRAPI_PTR* PFN_xrEnumerateXDevsMNDX)(XrXDevListMNDX xdevList,
                                                      uint32_t xdevCapacityInput,
                                                      uint32_t* xdevCountOutput,
                                                      XrXDevIdMNDX* xdevs);
typedef XrResult(XRAPI_PTR* PFN_xrGetXDevPropertiesMNDX)(XrXDevListMNDX xdevList,
                                                         const XrGetXDevInfoMNDX* info,
                                                         XrXDevPropertiesMNDX* properties);
typedef XrResult(XRAPI_PTR* PFN_xrDestroyXDevListMNDX)(XrXDevListMNDX xdevList);
