Description of a N frame history based temporal antialiasing system using a 'pingpong projection check'

Simplified description of the pingpong projection check:
- pixel position in current frame (pos1) + depth (d1)
- find world space for pos1 + d1 (ws1) (apply heading matrix to pos1, times d1, sum with cameraposition)
- find ws1 in history frame (pos2) + depth (d2) (apply history VPM to ws1)
- find worldspace for pos2 + d2 (ws2) (apply history heading matrix to pos2, times d2, sum with history cameraposition)
- find ws2 in current frame (pos3) (apply VPM)
- if pos1 is near pos3 allow contribution

Anti aliasing in 3d rendering is an attempt to avoid artifacts from limited sampling by either gathering more samples or applying post processing.

When rendering a 3d game at 60 frames a second, there is a large amount of closely related frames being generated.

What if we take the near duplicated work between frames and use that to increase the number of samples available per pixel, this is called temporal antialiasing.

Render every frame with a jittered offset, going through offsets in a cycle, a good jitter offset is an 8 queens/rooks, an offset used in dx msaa, or a halton sequence.

If the camera isn't moving and the scene isn't animated, averaging these frames together will output effectively the same as msaa/ssaa

However cameras move and then there is a lot of smearing.
To reduce this effect we can reproject our view vector into the older frames and find the sample there, however this gives issues if, due to parallax, it gatheres a sample that would be covered in our current frame.
So we take the depth at the point we sampled, turn it into a worldspace position and reproject it with our current view and if it hits our current pixel than it is usable, we can't use depth as a discard as depth differs a lot at the edges of object exactly where we need this information to have a good AA effect.

For the last N frames we keep a buffer of 3 linearcolorspace channels and 1 depth channel, the cameraposition, the jittered viewprojection matrix and heading matrix that turns the buffer position into a worldspace camera heading.

Then to generate the output we sample the depth of a pixel in the most recent frame, and the 4 neighbours, take the minimum, combine this with the unjittered worldspace heading for the pixel and the cameraposition to get a representative position in worldspace.

Then for each of the frames take the representative position and resolve to the frame position, discard if outside of clipspace.
At this frame position read the depth and combine with the frame heading matrix and frame camera position to find the reprojected terrain, find the output buffer position using the viewprojectionmatrix and if it is within clipspace and a 1 pixel radius, have it contribute to the output.
An accept radius of 1.2 has been suggested as giving better results.

Output depth is the minimum of contributed samples.
Color is average of contributed samples.

Negative input depth is used to skip sampling animated objects and textures or other elements that move too much.

Works well with screendoor transparancy if that too uses a jittered pattern.

Could be extended to use movement information.

The initial depth estimate uses an underbound, a range could be established and used for more precise culling of reprojections.

Ideally the whole history frames are forward projected, however this does not currently seem efficient on the gpu.

In my usage i use 8 frames of history.

Downsides:

All rendered frames are jittered, if insufficient samples are gathered, this jitter can show up both as temporal and positional noise and blurryness due to dejittering.
Does not handle animated textures nor moving elements in a scene.

Note:

Depth in this code is the linear worldspace distance from the camera to the hitpoint of the camera ray.


```hlsl
float3 BufferHeading(float4x4 headingMatrix, float2 coords) {
	float4 coord = (float4)0;
	coord.xy = coords.xy * Resolution;
	coord.w = 1;
	float4 projected = mul(headingMatrix, coord);
	projected.xyz /= projected.w;
	return normalize(projected.xyz);
}

inline float3 ScreenPos(float4x4 vpm, float4 position) {
    float4 pos = mul(vpm, position);
    float3 result;
    result.x = ((pos.x / pos.w) * 0.5 + 0.5);
    result.y = ((pos.y / pos.w) * -0.5 + 0.5);
    if (pos.w <= 0)
        result.z = -1;
    else
        result.z = pos.z / pos.w;
    return result;
}

inline void HandleTaaStep(inout float3 taa, inout float depth, inout float count, Texture2D<float4> source, float4x4 vpm, float4x4 historyHeadingMatrix, float4 historyCameraPos, float4 position, float2 screenPos) {
    float4 historyPosRaw = mul(vpm, position);
    float3 historyPos;
    historyPos.x = ((historyPosRaw.x / historyPosRaw.w) * 0.5 + 0.5);
    historyPos.y = ((historyPosRaw.y / historyPosRaw.w) * -0.5 + 0.5);
    historyPos.z = historyPosRaw.z / historyPosRaw.w;
    if ((historyPos.x <= 0.0) || (historyPos.y <= 0.0) || (historyPos.z <= 0.0) || (historyPos.x >= 1.0) || (historyPos.y >= 1.0)) {
        return;
    }
    float4 input = source.SampleLevel(LinearSampler, historyPos.xy, 0);
    float4 projectedPos;
    projectedPos.xyz = BufferHeading(historyHeadingMatrix, historyPos.xy) * input.a + historyCameraPos.xyz;
    projectedPos.w = 1;
    float3 reprojectedScreenPos = ScreenPos(ViewProjectionMatrix, projectedPos);
    float2 delta = abs(reprojectedScreenPos.xy * Resolution - screenPos.xy);
    if ((delta.x <= 1) && (delta.y <= 1) && (reprojectedScreenPos.z > 0.0))
    {
        taa.xyz += input.xyz;
        count += 1;
        depth = min(depth, input.a);
    }
}

[numthreads(64, 4, 1)]
void TaaMix(uint3 DTid : SV_DispatchThreadID)
{
    float4 base = taaHistory0[DTid.xy];
    float4 mixed = (float4)0;
    if (base.a <= 0) {
        mixed = base;
    }
    else
    {
        float depth = base.a;
        depth = min(depth, taaHistory0[clamp(DTid.xy + uint2(0, -1), float2(0, 0), UpperBufferLimit)].a);
        depth = min(depth, taaHistory0[clamp(DTid.xy + uint2(-1, 0), float2(0, 0), UpperBufferLimit)].a);
        depth = min(depth, taaHistory0[clamp(DTid.xy + uint2(1, 0), float2(0, 0), UpperBufferLimit)].a);
        depth = min(depth, taaHistory0[clamp(DTid.xy + uint2(0, 1), float2(0, 0), UpperBufferLimit)].a);

        float4 position;
        position.w = 1.0;
        position.xyz = BufferHeading(UnjitteredHeadingMatrix, DTid.xy * ResolutionInverted) * depth + CameraPosition.xyz;

        float3 taa = (float3)0;
        depth = base.a;
        float count = 0;
        float2 screenPos = DTid.xy;

        HandleTaaStep(taa, depth, count, taaHistory7, TaaHistoryVPM7, TaaHistoryVPM7Heading, TaaHistoryPosition7, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory6, TaaHistoryVPM6, TaaHistoryVPM6Heading, TaaHistoryPosition6, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory5, TaaHistoryVPM5, TaaHistoryVPM5Heading, TaaHistoryPosition5, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory4, TaaHistoryVPM4, TaaHistoryVPM4Heading, TaaHistoryPosition4, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory3, TaaHistoryVPM3, TaaHistoryVPM3Heading, TaaHistoryPosition3, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory2, TaaHistoryVPM2, TaaHistoryVPM2Heading, TaaHistoryPosition2, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory1, TaaHistoryVPM1, TaaHistoryVPM1Heading, TaaHistoryPosition1, position, screenPos);
        HandleTaaStep(taa, depth, count, taaHistory0, TaaHistoryVPM0, TaaHistoryVPM0Heading, TaaHistoryPosition0, position, screenPos);
        if (count == 0)
            mixed.rgb = base.rgb;
        else
            mixed.rgb = taa / count;

        mixed.a = depth;
    }
    mixedTexture[DTid.xy] = mixed;
}

```
