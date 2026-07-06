/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xr_plane_renderer_op.hpp"

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/quaternion.hpp"
#include "xdev_space_minimal.h"
#include "xr_hand_tracker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <sys/stat.h>
#include <utility>

namespace isaac_teleop::cam_streamer
{

namespace
{

template <size_t N>
std::string bounded_string(const char (&value)[N])
{
    const char* end = std::find(value, value + N, '\0');
    return std::string(value, end);
}

std::string ascii_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool contains_case_insensitive(const std::string& haystack, const char* needle)
{
    return ascii_lower(haystack).find(needle) != std::string::npos;
}

constexpr float kLeftHandColor[4] = { 0.10f, 0.82f, 1.00f, 1.00f };
constexpr float kRightHandColor[4] = { 1.00f, 0.55f, 0.18f, 1.00f };
constexpr float kHandLineWidthPx = 4.0f;
constexpr float kHandPointSizePx = 12.0f;
constexpr int32_t kHandLayerPriority = 10;
constexpr int32_t kStatusLayerPriority = 20;
constexpr int kStatusTextureWidth = 384;
constexpr int kStatusTextureHeight = 160;
constexpr float kStatusPanelDistanceM = 0.62f;
constexpr float kStatusPanelWidthM = 0.24f;
constexpr float kStatusPanelOffsetRightM = 0.20f;
constexpr float kStatusPanelOffsetUpM = 0.12f;
constexpr std::chrono::milliseconds kStatusPollInterval(150);

using HandBone = std::pair<xr::HandJointEXT, xr::HandJointEXT>;

struct ColorU8
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

constexpr std::array<HandBone, 22> kHandBones = {
    HandBone{ xr::HandJointEXT::Wrist, xr::HandJointEXT::Palm },
    HandBone{ xr::HandJointEXT::Palm, xr::HandJointEXT::ThumbMetacarpal },
    HandBone{ xr::HandJointEXT::ThumbMetacarpal, xr::HandJointEXT::ThumbProximal },
    HandBone{ xr::HandJointEXT::ThumbProximal, xr::HandJointEXT::ThumbDistal },
    HandBone{ xr::HandJointEXT::ThumbDistal, xr::HandJointEXT::ThumbTip },
    HandBone{ xr::HandJointEXT::Palm, xr::HandJointEXT::IndexMetacarpal },
    HandBone{ xr::HandJointEXT::IndexMetacarpal, xr::HandJointEXT::IndexProximal },
    HandBone{ xr::HandJointEXT::IndexProximal, xr::HandJointEXT::IndexIntermediate },
    HandBone{ xr::HandJointEXT::IndexIntermediate, xr::HandJointEXT::IndexDistal },
    HandBone{ xr::HandJointEXT::IndexDistal, xr::HandJointEXT::IndexTip },
    HandBone{ xr::HandJointEXT::Palm, xr::HandJointEXT::MiddleMetacarpal },
    HandBone{ xr::HandJointEXT::MiddleMetacarpal, xr::HandJointEXT::MiddleProximal },
    HandBone{ xr::HandJointEXT::MiddleProximal, xr::HandJointEXT::MiddleIntermediate },
    HandBone{ xr::HandJointEXT::MiddleIntermediate, xr::HandJointEXT::MiddleDistal },
    HandBone{ xr::HandJointEXT::MiddleDistal, xr::HandJointEXT::MiddleTip },
    HandBone{ xr::HandJointEXT::Palm, xr::HandJointEXT::RingMetacarpal },
    HandBone{ xr::HandJointEXT::RingMetacarpal, xr::HandJointEXT::RingProximal },
    HandBone{ xr::HandJointEXT::RingProximal, xr::HandJointEXT::RingIntermediate },
    HandBone{ xr::HandJointEXT::RingIntermediate, xr::HandJointEXT::RingDistal },
    HandBone{ xr::HandJointEXT::RingDistal, xr::HandJointEXT::RingTip },
    HandBone{ xr::HandJointEXT::Palm, xr::HandJointEXT::LittleMetacarpal },
    HandBone{ xr::HandJointEXT::LittleMetacarpal, xr::HandJointEXT::LittleProximal },
};

constexpr std::array<HandBone, 3> kLittleFingerTail = {
    HandBone{ xr::HandJointEXT::LittleProximal, xr::HandJointEXT::LittleIntermediate },
    HandBone{ xr::HandJointEXT::LittleIntermediate, xr::HandJointEXT::LittleDistal },
    HandBone{ xr::HandJointEXT::LittleDistal, xr::HandJointEXT::LittleTip },
};

constexpr std::array<HandBone, 3> kPalmCrossLinks = {
    HandBone{ xr::HandJointEXT::MiddleMetacarpal, xr::HandJointEXT::RingMetacarpal },
    HandBone{ xr::HandJointEXT::RingMetacarpal, xr::HandJointEXT::LittleMetacarpal },
    HandBone{ xr::HandJointEXT::ThumbMetacarpal, xr::HandJointEXT::IndexMetacarpal },
};

constexpr std::array<std::string_view, 26> kOpenXrHandJointNames = {
    "palm",        "wrist",           "thumb_metacarpal",   "thumb_proximal", "thumb_distal",
    "thumb_tip",   "index_metacarpal", "index_proximal",     "index_intermediate",
    "index_distal", "index_tip",       "middle_metacarpal",  "middle_proximal",
    "middle_intermediate",             "middle_distal",      "middle_tip",    "ring_metacarpal",
    "ring_proximal",                   "ring_intermediate",  "ring_distal",   "ring_tip",
    "little_metacarpal",               "little_proximal",    "little_intermediate",
    "little_distal",                   "little_tip",
};

// Get forward direction (-Z in OpenXR) from head orientation.
glm::vec3 get_forward(const xr::Quaternionf& orientation)
{
    glm::quat q(orientation.w, orientation.x, orientation.y, orientation.z);
    return q * glm::vec3(0.f, 0.f, -1.f);
}

// Convert xr::Vector3f to glm::vec3.
glm::vec3 to_glm(const xr::Vector3f& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

// Convert xr::Quaternionf to glm::quat.
glm::quat to_glm(const xr::Quaternionf& q)
{
    return glm::quat(q.w, q.x, q.y, q.z);
}

// Project forward direction onto XZ plane (horizontal).
glm::vec3 project_to_xz(const glm::vec3& forward)
{
    glm::vec3 forward_xz(forward.x, 0.f, forward.z);
    float len = glm::length(forward_xz);
    if (len > 0.001f)
    {
        return forward_xz / len;
    }
    return glm::vec3(0.f, 0.f, -1.f);
}

LockMode parse_lock_mode(const std::string& mode_str)
{
    if (mode_str == "world")
    {
        return LockMode::World;
    }
    else if (mode_str == "head")
    {
        return LockMode::Head;
    }
    return LockMode::Lazy; // Default
}

bool is_joint_position_valid(const xr::HandJointLocationEXT& joint)
{
    const auto flags = joint.locationFlags;
    return static_cast<bool>(flags & xr::SpaceLocationFlagBits::PositionValid) &&
           static_cast<bool>(flags & xr::SpaceLocationFlagBits::PositionTracked);
}

bool is_display_device_xdev(const XrXDevPropertiesMNDX& properties)
{
    const std::string name = bounded_string(properties.name);
    const std::string serial = bounded_string(properties.serial);
    return contains_case_insensitive(name, "displaydevice") || contains_case_insensitive(name, "display device") ||
           contains_case_insensitive(serial, "displaydevice") || contains_case_insensitive(serial, "display device") ||
           contains_case_insensitive(name, "head device") || contains_case_insensitive(serial, "head device");
}

uint64_t parse_env_uint64(const char* name, uint64_t fallback)
{
    const char* raw_value = std::getenv(name);
    if (raw_value == nullptr || raw_value[0] == '\0')
    {
        return fallback;
    }
    try
    {
        const auto parsed = std::stoull(raw_value);
        return parsed == 0 ? fallback : parsed;
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

void append_joint_xyz(const xr::Vector3f& point, std::vector<float>& vertices)
{
    vertices.push_back(point.x);
    vertices.push_back(point.y);
    vertices.push_back(point.z);
}

uint32_t append_hand_lines(const std::vector<xr::HandJointLocationEXT>& joints, std::vector<float>& vertices)
{
    uint32_t line_count = 0;
    auto append_bone = [&](xr::HandJointEXT from, xr::HandJointEXT to) {
        const auto from_index = static_cast<size_t>(from);
        const auto to_index = static_cast<size_t>(to);
        if (from_index >= joints.size() || to_index >= joints.size())
        {
            return;
        }

        const auto& start = joints[from_index];
        const auto& end = joints[to_index];
        if (!is_joint_position_valid(start) || !is_joint_position_valid(end))
        {
            return;
        }

        append_joint_xyz(start.pose.position, vertices);
        append_joint_xyz(end.pose.position, vertices);
        ++line_count;
    };

    for (const auto& bone : kHandBones)
    {
        append_bone(bone.first, bone.second);
    }
    for (const auto& bone : kLittleFingerTail)
    {
        append_bone(bone.first, bone.second);
    }
    for (const auto& bone : kPalmCrossLinks)
    {
        append_bone(bone.first, bone.second);
    }

    return line_count;
}

uint32_t append_hand_points(const std::vector<xr::HandJointLocationEXT>& joints, std::vector<float>& vertices)
{
    uint32_t point_count = 0;
    for (const auto& joint : joints)
    {
        if (!is_joint_position_valid(joint))
        {
            continue;
        }

        append_joint_xyz(joint.pose.position, vertices);
        ++point_count;
    }
    return point_count;
}

glm::mat4 make_eye_view_projection(const xr::CompositionLayerProjectionView& view,
                                   const xr::CompositionLayerDepthInfoKHR& depth_info)
{
    glm::mat4 view_rot = glm::mat4_cast(to_glm(view.pose.orientation));
    glm::mat4 view_trans = glm::translate(glm::mat4{ 1 }, glm::make_vec3(&view.pose.position.x));
    glm::mat4 view_mat = glm::inverse(view_trans * view_rot);

    const float nearZ = depth_info.nearZ;
    const float farZ = depth_info.farZ;
    glm::mat4 proj =
        glm::frustumRH_ZO(nearZ * glm::tan(view.fov.angleLeft), nearZ * glm::tan(view.fov.angleRight),
                          nearZ * glm::tan(view.fov.angleUp), nearZ * glm::tan(view.fov.angleDown), nearZ, farZ);

    return proj * view_mat;
}

bool find_json_value_start(const std::string& text, const std::string& key, size_t& value_pos)
{
    const std::string token = "\"" + key + "\"";
    const size_t key_pos = text.find(token);
    if (key_pos == std::string::npos)
    {
        return false;
    }
    const size_t colon_pos = text.find(':', key_pos + token.size());
    if (colon_pos == std::string::npos)
    {
        return false;
    }
    value_pos = colon_pos + 1;
    while (value_pos < text.size() && std::isspace(static_cast<unsigned char>(text[value_pos])))
    {
        ++value_pos;
    }
    return value_pos < text.size();
}

std::string parse_json_string_value(const std::string& text, const std::string& key)
{
    size_t value_pos = 0;
    if (!find_json_value_start(text, key, value_pos) || text[value_pos] != '"')
    {
        return {};
    }
    ++value_pos;
    const size_t end_pos = text.find('"', value_pos);
    if (end_pos == std::string::npos)
    {
        return {};
    }
    return text.substr(value_pos, end_pos - value_pos);
}

double parse_json_number_value(const std::string& text, const std::string& key, double fallback)
{
    size_t value_pos = 0;
    if (!find_json_value_start(text, key, value_pos))
    {
        return fallback;
    }
    size_t end_pos = value_pos;
    while (end_pos < text.size())
    {
        const char ch = text[end_pos];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E')
        {
            ++end_pos;
            continue;
        }
        break;
    }
    if (end_pos == value_pos)
    {
        return fallback;
    }
    try
    {
        return std::stod(text.substr(value_pos, end_pos - value_pos));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

ColorU8 parse_hex_color(const std::string& value, const ColorU8& fallback)
{
    if (value.size() != 7 || value[0] != '#')
    {
        return fallback;
    }
    try
    {
        const auto parse_component = [&](size_t offset) -> uint8_t {
            return static_cast<uint8_t>(std::stoi(value.substr(offset, 2), nullptr, 16));
        };
        return ColorU8{ parse_component(1), parse_component(3), parse_component(5), 255 };
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

size_t rgba_index(int x, int y, int width)
{
    return static_cast<size_t>((y * width + x) * 4);
}

void blend_pixel(std::vector<uint8_t>& rgba, int width, int height, int x, int y, const ColorU8& color)
{
    if (x < 0 || y < 0 || x >= width || y >= height || color.a == 0)
    {
        return;
    }

    const size_t idx = rgba_index(x, y, width);
    const uint8_t dst_r = rgba[idx + 0];
    const uint8_t dst_g = rgba[idx + 1];
    const uint8_t dst_b = rgba[idx + 2];
    const uint8_t dst_a = rgba[idx + 3];

    const float src_alpha = static_cast<float>(color.a) / 255.0f;
    const float dst_alpha = static_cast<float>(dst_a) / 255.0f;
    const float out_alpha = src_alpha + dst_alpha * (1.0f - src_alpha);
    if (out_alpha <= 0.0f)
    {
        return;
    }

    const auto blend_channel = [&](uint8_t src, uint8_t dst) -> uint8_t {
        const float src_value = static_cast<float>(src) / 255.0f;
        const float dst_value = static_cast<float>(dst) / 255.0f;
        const float out_value =
            (src_value * src_alpha + dst_value * dst_alpha * (1.0f - src_alpha)) / out_alpha;
        return static_cast<uint8_t>(std::round(std::clamp(out_value, 0.0f, 1.0f) * 255.0f));
    };

    rgba[idx + 0] = blend_channel(color.r, dst_r);
    rgba[idx + 1] = blend_channel(color.g, dst_g);
    rgba[idx + 2] = blend_channel(color.b, dst_b);
    rgba[idx + 3] = static_cast<uint8_t>(std::round(std::clamp(out_alpha, 0.0f, 1.0f) * 255.0f));
}

void fill_rect(std::vector<uint8_t>& rgba, int width, int height, int x, int y, int rect_width, int rect_height,
               const ColorU8& color)
{
    for (int row = 0; row < rect_height; ++row)
    {
        for (int col = 0; col < rect_width; ++col)
        {
            blend_pixel(rgba, width, height, x + col, y + row, color);
        }
    }
}

void fill_circle(std::vector<uint8_t>& rgba, int width, int height, int center_x, int center_y, int radius,
                 const ColorU8& color)
{
    const int radius_sq = radius * radius;
    for (int row = -radius; row <= radius; ++row)
    {
        for (int col = -radius; col <= radius; ++col)
        {
            if ((col * col + row * row) <= radius_sq)
            {
                blend_pixel(rgba, width, height, center_x + col, center_y + row, color);
            }
        }
    }
}

using GlyphPattern = std::array<const char*, 7>;

const GlyphPattern& glyph_for(char ch)
{
    static const GlyphPattern kSpace = { "     ", "     ", "     ", "     ", "     ", "     ", "     " };
    static const GlyphPattern kA = { " ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #" };
    static const GlyphPattern kD = { "#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### " };
    static const GlyphPattern kE = { "#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####" };
    static const GlyphPattern kF = { "#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    " };
    static const GlyphPattern kL = { "#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####" };
    static const GlyphPattern kN = { "#   #", "##  #", "##  #", "# # #", "#  ##", "#  ##", "#   #" };
    static const GlyphPattern kO = { " ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### " };
    static const GlyphPattern kP = { "#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    " };
    static const GlyphPattern kR = { "#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #" };
    static const GlyphPattern kS = { " ####", "#    ", "#    ", " ### ", "    #", "    #", "#### " };
    static const GlyphPattern kT = { "#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  " };
    static const GlyphPattern kU = { "#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### " };
    static const GlyphPattern kW = { "#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #" };
    static const GlyphPattern kY = { "#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  " };
    static const GlyphPattern kDash = { "     ", "     ", "     ", "#####", "     ", "     ", "     " };

    switch (ch)
    {
    case 'A':
        return kA;
    case 'D':
        return kD;
    case 'E':
        return kE;
    case 'F':
        return kF;
    case 'L':
        return kL;
    case 'N':
        return kN;
    case 'O':
        return kO;
    case 'P':
        return kP;
    case 'R':
        return kR;
    case 'S':
        return kS;
    case 'T':
        return kT;
    case 'U':
        return kU;
    case 'W':
        return kW;
    case 'Y':
        return kY;
    case '-':
        return kDash;
    case ' ':
    default:
        return kSpace;
    }
}

void draw_text(std::vector<uint8_t>& rgba, int width, int height, int x, int y, int scale, std::string_view text,
               const ColorU8& color)
{
    int cursor_x = x;
    for (char ch : text)
    {
        const auto& glyph = glyph_for(ch);
        for (int row = 0; row < static_cast<int>(glyph.size()); ++row)
        {
            const char* row_pattern = glyph[row];
            for (int col = 0; col < 5; ++col)
            {
                if (row_pattern[col] == ' ')
                {
                    continue;
                }
                fill_rect(rgba, width, height, cursor_x + col * scale, y + row * scale, scale, scale, color);
            }
        }
        cursor_x += 6 * scale;
    }
}

} // namespace

struct XDevHandTrackerSet
{
    std::shared_ptr<holoscan::XrSession> xr_session;
    XrXDevListMNDX xdev_list = XR_NULL_HANDLE;
    std::vector<XrHandTrackerEXT> left_trackers;
    std::vector<XrHandTrackerEXT> right_trackers;

    PFN_xrCreateHandTrackerEXT pfn_create_hand_tracker = nullptr;
    PFN_xrDestroyHandTrackerEXT pfn_destroy_hand_tracker = nullptr;
    PFN_xrLocateHandJointsEXT pfn_locate_hand_joints = nullptr;
    PFN_xrCreateXDevListMNDX pfn_create_xdev_list = nullptr;
    PFN_xrEnumerateXDevsMNDX pfn_enumerate_xdevs = nullptr;
    PFN_xrGetXDevPropertiesMNDX pfn_get_xdev_properties = nullptr;
    PFN_xrDestroyXDevListMNDX pfn_destroy_xdev_list = nullptr;

    ~XDevHandTrackerSet()
    {
        release();
    }

    bool load_function(const char* name, PFN_xrVoidFunction* ptr)
    {
        const XrResult result = xr_session->dispatch().xrGetInstanceProcAddr(xr_session->instance().get(), name, ptr);
        return XR_SUCCEEDED(result) && *ptr != nullptr;
    }

    bool initialize(std::shared_ptr<holoscan::XrSession> session)
    {
        xr_session = std::move(session);
        if (!xr_session)
        {
            return false;
        }

        if (!load_function("xrCreateHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_hand_tracker)) ||
            !load_function("xrDestroyHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_hand_tracker)) ||
            !load_function("xrLocateHandJointsEXT", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_locate_hand_joints)) ||
            !load_function("xrCreateXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_create_xdev_list)) ||
            !load_function("xrEnumerateXDevsMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_enumerate_xdevs)) ||
            !load_function("xrGetXDevPropertiesMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_get_xdev_properties)) ||
            !load_function("xrDestroyXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&pfn_destroy_xdev_list)))
        {
            return false;
        }

        XrCreateXDevListInfoMNDX create_info{ XR_TYPE_CREATE_XDEV_LIST_INFO_MNDX };
        XrResult result = pfn_create_xdev_list(xr_session->get().get(), &create_info, &xdev_list);
        if (XR_FAILED(result) || xdev_list == XR_NULL_HANDLE)
        {
            xdev_list = XR_NULL_HANDLE;
            return false;
        }

        uint32_t xdev_count = 0;
        result = pfn_enumerate_xdevs(xdev_list, 0, &xdev_count, nullptr);
        if (XR_FAILED(result) || xdev_count == 0)
        {
            release();
            return false;
        }

        std::vector<XrXDevIdMNDX> xdev_ids(xdev_count);
        result = pfn_enumerate_xdevs(xdev_list, xdev_count, &xdev_count, xdev_ids.data());
        if (XR_FAILED(result))
        {
            release();
            return false;
        }

        std::vector<XrXDevIdMNDX> preferred_xdev_ids;
        std::vector<XrXDevIdMNDX> display_xdev_ids;
        for (const XrXDevIdMNDX xdev_id : xdev_ids)
        {
            XrGetXDevInfoMNDX get_info{ XR_TYPE_GET_XDEV_INFO_MNDX };
            get_info.id = xdev_id;

            XrXDevPropertiesMNDX properties{ XR_TYPE_XDEV_PROPERTIES_MNDX };
            result = pfn_get_xdev_properties(xdev_list, &get_info, &properties);
            if (XR_FAILED(result))
            {
                continue;
            }

            HOLOSCAN_LOG_INFO("XrPlaneRendererOp: XDev id={} name='{}' serial='{}' canCreateSpace={}",
                              static_cast<uint64_t>(xdev_id), bounded_string(properties.name),
                              bounded_string(properties.serial), properties.canCreateSpace);
            if (is_display_device_xdev(properties))
            {
                display_xdev_ids.push_back(xdev_id);
            }
            else
            {
                preferred_xdev_ids.push_back(xdev_id);
            }
        }

        auto add_candidates = [this](const std::vector<XrXDevIdMNDX>& ids)
        {
            for (const XrXDevIdMNDX id : ids)
            {
                try_create_xdev_hand_tracker(id, XR_HAND_LEFT_EXT, left_trackers);
                try_create_xdev_hand_tracker(id, XR_HAND_RIGHT_EXT, right_trackers);
            }
        };
        add_candidates(preferred_xdev_ids);
        add_candidates(display_xdev_ids);

        if (left_trackers.empty() && right_trackers.empty())
        {
            release();
            return false;
        }

        HOLOSCAN_LOG_INFO("XrPlaneRendererOp: XDev hand trackers ready left_candidates={} right_candidates={}",
                          left_trackers.size(), right_trackers.size());
        return true;
    }

    bool try_create_xdev_hand_tracker(XrXDevIdMNDX xdev_id,
                                      XrHandEXT hand,
                                      std::vector<XrHandTrackerEXT>& trackers)
    {
        if (xdev_list == XR_NULL_HANDLE || xdev_id == 0 || pfn_create_hand_tracker == nullptr)
        {
            return false;
        }

        XrCreateHandTrackerXDevMNDX xdev_create_info{ XR_TYPE_CREATE_HAND_TRACKER_XDEV_MNDX };
        xdev_create_info.xdevList = xdev_list;
        xdev_create_info.id = xdev_id;

        XrHandTrackerCreateInfoEXT create_info{ XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT };
        create_info.next = &xdev_create_info;
        create_info.hand = hand;
        create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;

        XrHandTrackerEXT tracker = XR_NULL_HANDLE;
        const XrResult result = pfn_create_hand_tracker(xr_session->get().get(), &create_info, &tracker);
        if (XR_FAILED(result) || tracker == XR_NULL_HANDLE)
        {
            return false;
        }

        trackers.push_back(tracker);
        return true;
    }

    std::optional<std::vector<xr::HandJointLocationEXT>> locate_first_active(
        const std::vector<XrHandTrackerEXT>& trackers,
        xr::Time predicted_display_time)
    {
        if (pfn_locate_hand_joints == nullptr || !xr_session)
        {
            return {};
        }

        for (const XrHandTrackerEXT tracker : trackers)
        {
            if (tracker == XR_NULL_HANDLE)
            {
                continue;
            }

            XrHandJointsLocateInfoEXT locate_info{ XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT };
            locate_info.baseSpace = xr_session->reference_space().get();
            locate_info.time = predicted_display_time.get();

            std::vector<xr::HandJointLocationEXT> joint_data(XR_HAND_JOINT_COUNT_EXT);
            XrHandJointLocationsEXT locations{ XR_TYPE_HAND_JOINT_LOCATIONS_EXT };
            locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
            locations.jointLocations = reinterpret_cast<XrHandJointLocationEXT*>(joint_data.data());

            const XrResult result = pfn_locate_hand_joints(tracker, &locate_info, &locations);
            if (XR_FAILED(result) || !locations.isActive)
            {
                continue;
            }

            return joint_data;
        }
        return {};
    }

    void release()
    {
        if (pfn_destroy_hand_tracker != nullptr)
        {
            for (XrHandTrackerEXT& tracker : left_trackers)
            {
                if (tracker != XR_NULL_HANDLE)
                {
                    pfn_destroy_hand_tracker(tracker);
                    tracker = XR_NULL_HANDLE;
                }
            }
            for (XrHandTrackerEXT& tracker : right_trackers)
            {
                if (tracker != XR_NULL_HANDLE)
                {
                    pfn_destroy_hand_tracker(tracker);
                    tracker = XR_NULL_HANDLE;
                }
            }
        }
        left_trackers.clear();
        right_trackers.clear();

        if (xdev_list != XR_NULL_HANDLE && pfn_destroy_xdev_list != nullptr)
        {
            pfn_destroy_xdev_list(xdev_list);
            xdev_list = XR_NULL_HANDLE;
        }
    }
};

XrPlaneRendererOp::~XrPlaneRendererOp() = default;

void XrPlaneRendererOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<xr::FrameState>("xr_frame_state").condition(holoscan::ConditionType::kMessageAvailable);

    // Dynamic camera inputs - up to 8 planes (16 inputs for stereo)
    for (int i = 0; i < 8; i++)
    {
        std::string name_left = "camera_frame_" + std::to_string(i);
        std::string name_right = "camera_frame_" + std::to_string(i) + "_right";

        spec.input<holoscan::gxf::Entity>(name_left, holoscan::IOSpec::IOSize(1), holoscan::IOSpec::QueuePolicy::kPop)
            .condition(holoscan::ConditionType::kNone);
        spec.input<holoscan::gxf::Entity>(name_right, holoscan::IOSpec::IOSize(1), holoscan::IOSpec::QueuePolicy::kPop)
            .condition(holoscan::ConditionType::kNone);
    }

    spec.output<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");

    // Parameters (plane configs are set via set_plane_configs() before initialize)
    spec.param(xr_session_, "xr_session", "XR Session", "OpenXR session");
    spec.param(left_hand_tracker_, "left_hand_tracker", "Left Hand Tracker", "OpenXR left hand tracker",
               std::shared_ptr<holoscan::XrHandTracker>{});
    spec.param(right_hand_tracker_, "right_hand_tracker", "Right Hand Tracker", "OpenXR right hand tracker",
               std::shared_ptr<holoscan::XrHandTracker>{});
    spec.param(verbose_, "verbose", "Verbose", "Enable verbose logging", false);

    cuda_stream_handler_.define_params(spec);
}

void XrPlaneRendererOp::initialize()
{
    Operator::initialize();

    // Use plane configs set via set_plane_configs()
    if (plane_configs_.empty())
    {
        HOLOSCAN_LOG_ERROR("XrPlaneRendererOp: No planes configured. Call set_plane_configs() first.");
        return;
    }

    // Build plane states from configs
    planes_.resize(plane_configs_.size());
    for (size_t i = 0; i < plane_configs_.size(); i++)
    {
        planes_[i].config = plane_configs_[i];
        planes_[i].input_index = i;
    }

    // Sort planes by distance (farthest first) for proper depth rendering
    std::sort(planes_.begin(), planes_.end(),
              [](const PlaneState& a, const PlaneState& b) { return a.config.distance > b.config.distance; });

    if (verbose_.get())
    {
        HOLOSCAN_LOG_INFO("XrPlaneRendererOp: {} planes configured", planes_.size());
        for (const auto& plane : planes_)
        {
            HOLOSCAN_LOG_INFO("  - {}: distance={}m, width={}m, offset=({}, {})m, stereo={}", plane.config.name,
                              plane.config.distance, plane.config.width, plane.config.offset_x, plane.config.offset_y,
                              plane.config.is_stereo ? "true" : "false");
        }

        HOLOSCAN_LOG_INFO("XrPlaneRendererOp: hand skeleton overlay {}", left_hand_tracker_.get() && right_hand_tracker_.get()
                                                                          ? "enabled"
                                                                          : "disabled");
    }
}

void XrPlaneRendererOp::start()
{
    auto xr_session = xr_session_.get();
    uint32_t width =
        xr_session->view_configurations()[0].recommendedImageRectWidth * xr_session->view_configurations().size();
    uint32_t height = xr_session->view_configurations()[0].recommendedImageRectHeight;

    holoviz_instance_ = holoscan::viz::Create();
    holoscan::viz::SetCurrent(holoviz_instance_);
    holoscan::viz::Init(width, height, "XR Multi Plane", holoscan::viz::InitFlags::HEADLESS);

    color_swapchain_ = std::make_unique<holoscan::XrSwapchainCuda>(
        *xr_session, holoscan::XrSwapchainCuda::Format::R8G8B8A8_SRGB, width, height);
    depth_swapchain_ = std::make_unique<holoscan::XrSwapchainCuda>(
        *xr_session, holoscan::XrSwapchainCuda::Format::D32_SFLOAT, width, height);

    // Initialize the main tracker - this drives lazy locking for all planes
    // The main plane is the first one in the config (usually head camera)
    if (!planes_.empty())
    {
        auto& main_plane = planes_[0];
        CameraPlaneConfig config;
        config.lock_mode = parse_lock_mode(main_plane.config.lock_mode);
        config.distance = main_plane.config.distance;
        config.width = main_plane.config.width;
        config.offset_x = 0.0f; // Main plane has no offset
        config.offset_y = 0.0f;
        config.look_away_angle = main_plane.config.look_away_angle;
        config.reposition_distance = main_plane.config.reposition_distance;
        config.reposition_delay = main_plane.config.reposition_delay;
        config.transition_duration = main_plane.config.transition_duration;
        main_tracker_ = std::make_unique<CameraPlane>(config);
    }

    if (verbose_.get())
    {
        HOLOSCAN_LOG_INFO("XrPlaneRendererOp started: {}x{}, {} planes", width, height, planes_.size());
    }

    const char* status_path = std::getenv("TELEOP_XR_STATUS_PATH");
    teleop_status_path_ = status_path != nullptr ? std::string(status_path) : std::string();
    last_status_poll_ = std::chrono::steady_clock::time_point{};
    status_overlay_rgba_.assign(static_cast<size_t>(kStatusTextureWidth * kStatusTextureHeight * 4), 0);
    status_overlay_bytes_ = status_overlay_rgba_.size();
    status_overlay_dirty_ = true;

    if (!teleop_status_path_.empty() && verbose_.get())
    {
        HOLOSCAN_LOG_INFO("XrPlaneRendererOp: teleop status overlay path={}", teleop_status_path_);
    }

    const char* hand_log_path = std::getenv("TELEOP_XR_HAND_LOG_PATH");
    hand_log_path_ = hand_log_path != nullptr ? std::string(hand_log_path) : std::string();
    hand_log_stride_ = parse_env_uint64("TELEOP_XR_HAND_LOG_STRIDE", 10);
    if (!hand_log_path_.empty())
    {
        hand_log_.open(hand_log_path_, std::ios::out | std::ios::app);
        if (!hand_log_.good())
        {
            HOLOSCAN_LOG_WARN("XrPlaneRendererOp: failed to open XR hand log path={}", hand_log_path_);
        }
        else
        {
            chmod(hand_log_path_.c_str(), 0666);
            HOLOSCAN_LOG_INFO("XrPlaneRendererOp: XR hand log path={} stride={}", hand_log_path_, hand_log_stride_);
        }
    }

    initialize_xdev_hand_tracking();
}

void XrPlaneRendererOp::stop()
{
    for (auto& plane : planes_)
    {
        plane.entity_left = holoscan::gxf::Entity();
        plane.entity_right = holoscan::gxf::Entity();
        plane.data_left = nullptr;
        plane.data_right = nullptr;
    }
    main_tracker_.reset();
    release_status_overlay();

    color_swapchain_.reset();
    depth_swapchain_.reset();

    holoscan::viz::Shutdown(holoviz_instance_);
    holoviz_instance_ = nullptr;

    if (verbose_.get())
    {
        HOLOSCAN_LOG_INFO("XrPlaneRendererOp stopped. Frames: {}", frame_count_);
    }
    if (hand_log_.is_open())
    {
        hand_log_.close();
    }
    release_xdev_hand_tracking();
}

void XrPlaneRendererOp::initialize_xdev_hand_tracking()
{
    xdev_hand_tracker_set_ = std::make_shared<XDevHandTrackerSet>();
    if (!xdev_hand_tracker_set_->initialize(xr_session_.get()))
    {
        xdev_hand_tracker_set_.reset();
        HOLOSCAN_LOG_WARN("XrPlaneRendererOp: XDev hand trackers unavailable; using default hand trackers only");
    }
}

void XrPlaneRendererOp::release_xdev_hand_tracking()
{
    xdev_hand_tracker_set_.reset();
}

void XrPlaneRendererOp::compute(holoscan::InputContext& input,
                                holoscan::OutputContext& output,
                                holoscan::ExecutionContext& context)
{
    auto xr_session = xr_session_.get();
    auto frame_state = input.receive<xr::FrameState>("xr_frame_state");
    current_frame_state_ = *frame_state;

    // Update camera frames for each plane
    for (size_t i = 0; i < planes_.size(); i++)
    {
        auto& plane = planes_[i];
        std::string input_left = "camera_frame_" + std::to_string(plane.input_index);
        std::string input_right = "camera_frame_" + std::to_string(plane.input_index) + "_right";

        // Update left/mono frame
        if (!input.empty(input_left.c_str()))
        {
            auto entity = input.receive<holoscan::gxf::Entity>(input_left.c_str());
            if (entity)
            {
                auto& gxf_entity = static_cast<nvidia::gxf::Entity&>(entity.value());
                auto tensor = gxf_entity.get<nvidia::gxf::Tensor>();
                if (tensor)
                {
                    plane.entity_left = entity.value();
                    plane.data_left = tensor.value()->pointer();
                    plane.width_left = tensor.value()->shape().dimension(1);
                    plane.height_left = tensor.value()->shape().dimension(0);
                }
            }
        }

        // Update right frame (stereo mode)
        if (plane.config.is_stereo && !input.empty(input_right.c_str()))
        {
            auto entity = input.receive<holoscan::gxf::Entity>(input_right.c_str());
            if (entity)
            {
                auto& gxf_entity = static_cast<nvidia::gxf::Entity&>(entity.value());
                auto tensor = gxf_entity.get<nvidia::gxf::Tensor>();
                if (tensor)
                {
                    plane.entity_right = entity.value();
                    plane.data_right = tensor.value()->pointer();
                    plane.width_right = tensor.value()->shape().dimension(1);
                    plane.height_right = tensor.value()->shape().dimension(0);
                }
            }
        }
    }

    // Check if any plane has data
    bool has_any_data = false;
    for (const auto& plane : planes_)
    {
        if (plane.has_data())
        {
            has_any_data = true;
            break;
        }
    }

    if (!has_any_data)
    {
        output.emit(std::shared_ptr<xr::CompositionLayerBaseHeader>(nullptr), "xr_composition_layer");
        return;
    }

    // Get head pose and update main tracker only
    // Secondary planes will be positioned relative to the main plane
    xr::Space reference_space = xr_session->reference_space();
    xr::SpaceLocation head_location =
        xr_session->view_space().locateSpace(reference_space, frame_state->predictedDisplayTime);

    glm::vec3 head_pos = to_glm(head_location.pose.position);
    glm::quat head_orientation = to_glm(head_location.pose.orientation);
    glm::vec3 forward_xz = project_to_xz(get_forward(head_location.pose.orientation));

    if (main_tracker_)
    {
        main_tracker_->update(head_pos, head_orientation, forward_xz);
    }

    // Create composition layer
    auto composition_layer = holoscan::XrCompositionLayerProjectionStorage::create_for_frame(
        *frame_state, *xr_session, *color_swapchain_, *depth_swapchain_);

    auto color_tensor = color_swapchain_->acquire();
    auto depth_tensor = depth_swapchain_->acquire();
    current_cuda_stream_ = cuda_stream_handler_.get_cuda_stream(context.context());

    holoscan::viz::SetCurrent(holoviz_instance_);
    holoscan::viz::SetCudaStream(current_cuda_stream_);

    // Render all planes within a single render pass
    render_planes(composition_layer, head_pos);
    render_hand_overlays(composition_layer);
    render_status_overlay(composition_layer, head_pos, head_orientation);

    // Read back the framebuffer
    holoscan::viz::ReadFramebuffer(holoscan::viz::ImageFormat::R8G8B8A8_UNORM, color_swapchain_->width(),
                                   color_swapchain_->height(), color_tensor.nbytes(),
                                   reinterpret_cast<CUdeviceptr>(color_tensor.data()));
    holoscan::viz::ReadFramebuffer(holoscan::viz::ImageFormat::D32_SFLOAT, depth_swapchain_->width(),
                                   depth_swapchain_->height(), depth_tensor.nbytes(),
                                   reinterpret_cast<CUdeviceptr>(depth_tensor.data()));

    color_swapchain_->release(current_cuda_stream_);
    depth_swapchain_->release(current_cuda_stream_);

    frame_count_++;
    output.emit(std::static_pointer_cast<xr::CompositionLayerBaseHeader>(composition_layer), "xr_composition_layer");
}

void XrPlaneRendererOp::render_planes(const std::shared_ptr<holoscan::XrCompositionLayerProjectionStorage>& layer,
                                      const glm::vec3& head_pos)
{

    if (!main_tracker_)
        return;

    // Get main plane position and rotation - this drives all planes
    glm::vec3 main_pos = main_tracker_->position();
    glm::quat main_rotation = main_tracker_->rotation();

    // Render all planes for all eyes in a single Begin/End block
    holoscan::viz::Begin();

    // Render all planes (sorted by distance, farthest first)
    for (size_t plane_idx = 0; plane_idx < planes_.size(); plane_idx++)
    {
        auto& plane = planes_[plane_idx];
        if (!plane.has_data())
            continue;

        // Compute plane position relative to main plane
        glm::vec3 plane_pos;
        glm::quat plane_rotation;

        if (plane_idx == 0)
        {
            // Main plane - use main tracker position and rotation directly
            plane_pos = main_pos;
            plane_rotation = main_rotation;
        }
        else
        {
            // Secondary plane - positioned relative to the main plane
            // This ensures secondary planes stay anchored when main plane is lazy-locked

            // Get forward/right/up directions from main plane's orientation
            glm::vec3 forward = main_rotation * glm::vec3(0.0f, 0.0f, -1.0f);
            glm::vec3 right = main_rotation * glm::vec3(1.0f, 0.0f, 0.0f);
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f); // World up

            // Offset from main plane position:
            // - Forward: difference in distance from main plane's distance
            // - Right/Up: configured offsets
            float main_distance = planes_[0].config.distance;
            float distance_offset = plane.config.distance - main_distance;

            plane_pos = main_pos + forward * distance_offset + right * plane.config.offset_x + up * plane.config.offset_y;

            // Handle rotation based on transition state:
            // - During transition: face the user (compute dynamically)
            // - After transition: use locked rotation
            if (main_tracker_->is_transitioning())
            {
                // Transitioning - compute rotation to face user
                plane.rotation_locked = false;

                glm::vec3 to_head = head_pos - plane_pos;
                to_head.y = 0.0f; // Project to horizontal plane
                float len = glm::length(to_head);
                if (len > 0.001f)
                {
                    to_head /= len;
                    float yaw = std::atan2(to_head.x, to_head.z);
                    plane_rotation = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
                }
                else
                {
                    plane_rotation = main_rotation;
                }
            }
            else
            {
                // Not transitioning - lock/use locked rotation
                if (!plane.rotation_locked)
                {
                    // Just finished transitioning - lock current facing rotation
                    glm::vec3 to_head = head_pos - plane_pos;
                    to_head.y = 0.0f;
                    float len = glm::length(to_head);
                    if (len > 0.001f)
                    {
                        to_head /= len;
                        float yaw = std::atan2(to_head.x, to_head.z);
                        plane.locked_rotation = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
                    }
                    else
                    {
                        plane.locked_rotation = main_rotation;
                    }
                    plane.rotation_locked = true;
                }
                plane_rotation = plane.locked_rotation;
            }
        }

        for (int eye_idx = 0; eye_idx < layer->viewCount; eye_idx++)
        {
            auto& view = layer->views[eye_idx];

            // For stereo planes, use right-eye data for eye_idx 1.
            // Falls back to data_left when data_right is unavailable (the else
            // branch covers both mono planes and missing right buffers).
            const void* frame_data;
            int frame_width;
            int frame_height;
            if (plane.config.is_stereo && eye_idx == 1 && plane.data_right)
            {
                frame_data = plane.data_right;
                frame_width = plane.width_right;
                frame_height = plane.height_right;
            }
            else
            {
                frame_data = plane.data_left;
                frame_width = plane.width_left;
                frame_height = plane.height_left;
            }
            if (!frame_data)
                continue;

            float aspect = static_cast<float>(frame_height) / static_cast<float>(frame_width);

            holoscan::viz::BeginImageLayer();

            holoscan::viz::ImageCudaDevice(frame_width, frame_height, holoscan::viz::ImageFormat::R8G8B8_UNORM,
                                           reinterpret_cast<CUdeviceptr>(frame_data));

            // Compute MVP for this plane and eye
            glm::mat4 model = glm::translate(glm::mat4{ 1 }, plane_pos);
            model = model * glm::mat4_cast(plane_rotation);
            model = glm::scale(model, glm::vec3(plane.config.width, -plane.config.width * aspect, 1.f));

            glm::mat4 eye_view_projection = make_eye_view_projection(view, layer->depth_info[eye_idx]);
            glm::mat4 mvp = glm::transpose(eye_view_projection * model);

            // Add view for just this eye's region
            holoscan::viz::LayerAddView(
                static_cast<float>(view.subImage.imageRect.offset.x) / color_swapchain_->width(),
                static_cast<float>(view.subImage.imageRect.offset.y) / color_swapchain_->height(),
                static_cast<float>(view.subImage.imageRect.extent.width) / color_swapchain_->width(),
                static_cast<float>(view.subImage.imageRect.extent.height) / color_swapchain_->height(),
                glm::value_ptr(mvp));

            holoscan::viz::EndLayer();
        }
    }

    holoscan::viz::End();
}

void XrPlaneRendererOp::refresh_status_overlay()
{
    if (teleop_status_path_.empty())
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (last_status_poll_.time_since_epoch().count() != 0 &&
        (now - last_status_poll_) < kStatusPollInterval && !status_overlay_dirty_)
    {
        return;
    }
    last_status_poll_ = now;

    std::ifstream input(teleop_status_path_);
    if (!input.good())
    {
        if (teleop_status_.has_content())
        {
            teleop_status_ = TeleopStatusOverlay{};
            status_overlay_dirty_ = true;
        }
        return;
    }

    std::stringstream buffer;
    buffer << input.rdbuf();
    const std::string payload = buffer.str();

    TeleopStatusOverlay next_status;
    next_status.badge_state = parse_json_string_value(payload, "badge_state");
    next_status.badge_label = parse_json_string_value(payload, "badge_label");
    next_status.badge_color = parse_json_string_value(payload, "badge_color");
    next_status.toast_label = parse_json_string_value(payload, "toast_label");
    next_status.toast_color = parse_json_string_value(payload, "toast_color");
    next_status.toast_until_s = parse_json_number_value(payload, "toast_until_s", 0.0);
    next_status.updated_at_s = parse_json_number_value(payload, "updated_at_s", 0.0);

    if (next_status.badge_label != teleop_status_.badge_label || next_status.badge_color != teleop_status_.badge_color ||
        next_status.toast_label != teleop_status_.toast_label || next_status.toast_color != teleop_status_.toast_color ||
        next_status.toast_until_s != teleop_status_.toast_until_s ||
        next_status.badge_state != teleop_status_.badge_state)
    {
        teleop_status_ = std::move(next_status);
        status_overlay_dirty_ = true;
    }
}

void XrPlaneRendererOp::rebuild_status_overlay_texture()
{
    if (!teleop_status_.has_content())
    {
        status_overlay_rgba_.assign(status_overlay_rgba_.size(), 0);
    }
    else
    {
        std::fill(status_overlay_rgba_.begin(), status_overlay_rgba_.end(), 0);

        const ColorU8 card_bg{ 14, 18, 24, 212 };
        const ColorU8 card_bg_secondary{ 14, 18, 24, 196 };
        const ColorU8 white{ 255, 255, 255, 255 };
        const ColorU8 badge_color = parse_hex_color(teleop_status_.badge_color, ColorU8{ 122, 128, 140, 255 });
        const ColorU8 toast_color = parse_hex_color(teleop_status_.toast_color, badge_color);

        fill_rect(status_overlay_rgba_, kStatusTextureWidth, kStatusTextureHeight, 12, 14, 224, 58, card_bg);
        fill_circle(status_overlay_rgba_, kStatusTextureWidth, kStatusTextureHeight, 36, 43, 10, badge_color);
        draw_text(status_overlay_rgba_, kStatusTextureWidth, kStatusTextureHeight, 56, 22, 4,
                  teleop_status_.badge_label, white);

        const double now_wall_s =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (!teleop_status_.toast_label.empty() && now_wall_s < teleop_status_.toast_until_s)
        {
            fill_rect(status_overlay_rgba_, kStatusTextureWidth, kStatusTextureHeight, 12, 86, 188, 44,
                      card_bg_secondary);
            fill_rect(status_overlay_rgba_, kStatusTextureWidth, kStatusTextureHeight, 12, 86, 6, 44, toast_color);
            draw_text(status_overlay_rgba_, kStatusTextureWidth, kStatusTextureHeight, 30, 96, 3,
                      teleop_status_.toast_label, white);
        }
    }

    if (status_overlay_device_ == nullptr)
    {
        const auto cuda_result = cudaMalloc(reinterpret_cast<void**>(&status_overlay_device_), status_overlay_bytes_);
        if (cuda_result != cudaSuccess)
        {
            HOLOSCAN_LOG_WARN("XrPlaneRendererOp: failed to allocate XR status overlay texture: {}",
                              cudaGetErrorString(cuda_result));
            status_overlay_device_ = nullptr;
            return;
        }
    }

    const auto copy_result =
        cudaMemcpyAsync(status_overlay_device_, status_overlay_rgba_.data(), status_overlay_bytes_,
                        cudaMemcpyHostToDevice, current_cuda_stream_);
    if (copy_result != cudaSuccess)
    {
        HOLOSCAN_LOG_WARN("XrPlaneRendererOp: failed to upload XR status overlay texture: {}",
                          cudaGetErrorString(copy_result));
        return;
    }
    status_overlay_dirty_ = false;
}

void XrPlaneRendererOp::release_status_overlay()
{
    if (status_overlay_device_ != nullptr)
    {
        cudaFree(status_overlay_device_);
        status_overlay_device_ = nullptr;
    }
    status_overlay_rgba_.clear();
    status_overlay_bytes_ = 0;
    teleop_status_ = TeleopStatusOverlay{};
    status_overlay_dirty_ = false;
}

void XrPlaneRendererOp::append_hand_log(const char* hand_name,
                                        const std::vector<xr::HandJointLocationEXT>& joints,
                                        uint32_t valid_point_count)
{
    if (!hand_log_.is_open() || !hand_log_.good())
    {
        return;
    }
    if (hand_log_stride_ > 1 && (frame_count_ % hand_log_stride_) != 0)
    {
        return;
    }
    if (joints.size() < kOpenXrHandJointNames.size() || valid_point_count == 0)
    {
        return;
    }

    const double unix_time_s =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const double monotonic_time_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

    hand_log_ << std::setprecision(9)
              << "{\"schema_version\":\"teleop_stack.xr_hand_sample_stream.v1\","
              << "\"event\":\"frame\","
              << "\"source\":\"camera_streamer_overlay\","
              << "\"hand\":\"" << hand_name << "\","
              << "\"frame_index\":" << frame_count_ << ","
              << "\"time_s\":" << unix_time_s << ","
              << "\"monotonic_time_s\":" << monotonic_time_s << ","
              << "\"valid_joint_count\":" << valid_point_count << ",";

    hand_log_ << "\"raw_hand_joint_names\":[";
    for (size_t i = 0; i < kOpenXrHandJointNames.size(); ++i)
    {
        if (i > 0)
        {
            hand_log_ << ",";
        }
        hand_log_ << "\"" << kOpenXrHandJointNames[i] << "\"";
    }
    hand_log_ << "],";

    hand_log_ << "\"joint_valid\":[";
    for (size_t i = 0; i < kOpenXrHandJointNames.size(); ++i)
    {
        if (i > 0)
        {
            hand_log_ << ",";
        }
        hand_log_ << (is_joint_position_valid(joints[i]) ? "true" : "false");
    }
    hand_log_ << "],";

    hand_log_ << "\"raw_hand_positions_xyz\":[";
    for (size_t i = 0; i < kOpenXrHandJointNames.size(); ++i)
    {
        if (i > 0)
        {
            hand_log_ << ",";
        }
        const auto& position = joints[i].pose.position;
        hand_log_ << "[" << position.x << "," << position.y << "," << position.z << "]";
    }
    hand_log_ << "],";

    hand_log_ << "\"raw_hand_orientations_xyzw\":[";
    for (size_t i = 0; i < kOpenXrHandJointNames.size(); ++i)
    {
        if (i > 0)
        {
            hand_log_ << ",";
        }
        const auto& orientation = joints[i].pose.orientation;
        hand_log_ << "[" << orientation.x << "," << orientation.y << "," << orientation.z << "," << orientation.w
                  << "]";
    }
    hand_log_ << "]}\n";
    hand_log_.flush();
}

void XrPlaneRendererOp::render_status_overlay(
    const std::shared_ptr<holoscan::XrCompositionLayerProjectionStorage>& layer,
    const glm::vec3& head_pos,
    const glm::quat& head_orientation)
{
    refresh_status_overlay();
    if (!teleop_status_.has_content())
    {
        return;
    }
    if (status_overlay_device_ == nullptr || status_overlay_dirty_)
    {
        rebuild_status_overlay_texture();
    }
    if (status_overlay_device_ == nullptr)
    {
        return;
    }

    const glm::vec3 forward = head_orientation * glm::vec3(0.0f, 0.0f, -1.0f);
    const glm::vec3 right = head_orientation * glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 up = head_orientation * glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 overlay_pos = head_pos + forward * kStatusPanelDistanceM + right * kStatusPanelOffsetRightM +
                                  up * kStatusPanelOffsetUpM;
    const float aspect = static_cast<float>(kStatusTextureHeight) / static_cast<float>(kStatusTextureWidth);

    for (int eye_idx = 0; eye_idx < layer->viewCount; eye_idx++)
    {
        const auto& view = layer->views[eye_idx];

        holoscan::viz::BeginImageLayer();
        holoscan::viz::LayerPriority(kStatusLayerPriority);
        holoscan::viz::ImageCudaDevice(kStatusTextureWidth, kStatusTextureHeight,
                                       holoscan::viz::ImageFormat::R8G8B8A8_UNORM,
                                       reinterpret_cast<CUdeviceptr>(status_overlay_device_));

        glm::mat4 model = glm::translate(glm::mat4{ 1 }, overlay_pos);
        model = model * glm::mat4_cast(head_orientation);
        model = glm::scale(model, glm::vec3(kStatusPanelWidthM, -kStatusPanelWidthM * aspect, 1.0f));

        const glm::mat4 eye_view_projection = make_eye_view_projection(view, layer->depth_info[eye_idx]);
        const glm::mat4 mvp = glm::transpose(eye_view_projection * model);

        holoscan::viz::LayerAddView(
            static_cast<float>(view.subImage.imageRect.offset.x) / color_swapchain_->width(),
            static_cast<float>(view.subImage.imageRect.offset.y) / color_swapchain_->height(),
            static_cast<float>(view.subImage.imageRect.extent.width) / color_swapchain_->width(),
            static_cast<float>(view.subImage.imageRect.extent.height) / color_swapchain_->height(),
            glm::value_ptr(mvp));
        holoscan::viz::EndLayer();
    }
}

void XrPlaneRendererOp::render_hand_overlays(
    const std::shared_ptr<holoscan::XrCompositionLayerProjectionStorage>& layer)
{
    auto left_hand_tracker = left_hand_tracker_.get();
    auto right_hand_tracker = right_hand_tracker_.get();
    if (!left_hand_tracker || !right_hand_tracker)
    {
        return;
    }

    std::optional<std::vector<xr::HandJointLocationEXT>> left_hand;
    std::optional<std::vector<xr::HandJointLocationEXT>> right_hand;
    if (xdev_hand_tracker_set_)
    {
        left_hand = xdev_hand_tracker_set_->locate_first_active(xdev_hand_tracker_set_->left_trackers,
                                                                current_frame_state_.predictedDisplayTime);
        right_hand = xdev_hand_tracker_set_->locate_first_active(xdev_hand_tracker_set_->right_trackers,
                                                                 current_frame_state_.predictedDisplayTime);
    }
    if (!left_hand.has_value())
    {
        left_hand = left_hand_tracker->locate_hand_joints();
    }
    if (!right_hand.has_value())
    {
        right_hand = right_hand_tracker->locate_hand_joints();
    }
    if (!left_hand.has_value() && !right_hand.has_value())
    {
        return;
    }

    std::vector<float> left_line_vertices;
    std::vector<float> right_line_vertices;
    std::vector<float> left_point_vertices;
    std::vector<float> right_point_vertices;

    uint32_t left_line_count = 0;
    uint32_t right_line_count = 0;
    uint32_t left_point_count = 0;
    uint32_t right_point_count = 0;

    if (left_hand.has_value())
    {
        left_line_count = append_hand_lines(left_hand.value(), left_line_vertices);
        left_point_count = append_hand_points(left_hand.value(), left_point_vertices);
    }
    if (right_hand.has_value())
    {
        right_line_count = append_hand_lines(right_hand.value(), right_line_vertices);
        right_point_count = append_hand_points(right_hand.value(), right_point_vertices);
    }

    if (left_hand.has_value())
    {
        append_hand_log("left", left_hand.value(), left_point_count);
    }
    if (right_hand.has_value())
    {
        append_hand_log("right", right_hand.value(), right_point_count);
    }

    if ((left_line_count + right_line_count + left_point_count + right_point_count) == 0)
    {
        return;
    }

    for (int eye_idx = 0; eye_idx < layer->viewCount; eye_idx++)
    {
        const auto& view = layer->views[eye_idx];
        const glm::mat4 view_projection = make_eye_view_projection(view, layer->depth_info[eye_idx]);

        holoscan::viz::BeginGeometryLayer();
        holoscan::viz::LayerPriority(kHandLayerPriority);
        holoscan::viz::LineWidth(kHandLineWidthPx);
        holoscan::viz::PointSize(kHandPointSizePx);

        if (left_line_count > 0)
        {
            holoscan::viz::Color(kLeftHandColor[0], kLeftHandColor[1], kLeftHandColor[2], kLeftHandColor[3]);
            holoscan::viz::Primitive(holoscan::viz::PrimitiveTopology::LINE_LIST_3D, left_line_count,
                                     left_line_vertices.size(), left_line_vertices.data());
        }
        if (right_line_count > 0)
        {
            holoscan::viz::Color(kRightHandColor[0], kRightHandColor[1], kRightHandColor[2], kRightHandColor[3]);
            holoscan::viz::Primitive(holoscan::viz::PrimitiveTopology::LINE_LIST_3D, right_line_count,
                                     right_line_vertices.size(), right_line_vertices.data());
        }
        if (left_point_count > 0)
        {
            holoscan::viz::Color(kLeftHandColor[0], kLeftHandColor[1], kLeftHandColor[2], kLeftHandColor[3]);
            holoscan::viz::Primitive(holoscan::viz::PrimitiveTopology::POINT_LIST_3D, left_point_count,
                                     left_point_vertices.size(), left_point_vertices.data());
        }
        if (right_point_count > 0)
        {
            holoscan::viz::Color(kRightHandColor[0], kRightHandColor[1], kRightHandColor[2], kRightHandColor[3]);
            holoscan::viz::Primitive(holoscan::viz::PrimitiveTopology::POINT_LIST_3D, right_point_count,
                                     right_point_vertices.size(), right_point_vertices.data());
        }

        holoscan::viz::LayerAddView(
            static_cast<float>(view.subImage.imageRect.offset.x) / color_swapchain_->width(),
            static_cast<float>(view.subImage.imageRect.offset.y) / color_swapchain_->height(),
            static_cast<float>(view.subImage.imageRect.extent.width) / color_swapchain_->width(),
            static_cast<float>(view.subImage.imageRect.extent.height) / color_swapchain_->height(),
            glm::value_ptr(glm::transpose(view_projection)));
        holoscan::viz::EndLayer();
    }
}

} // namespace isaac_teleop::cam_streamer
