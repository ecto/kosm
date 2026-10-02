// Partial sums: component-major data[c * n + i] → out[c * groups + g], each
// workgroup summing 256 consecutive elements of one component.

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = wid.x * 256u + lid.x;
    let c = wid.y;
    var v = 0.0;
    if (i < p.n) {
        v = data[c * p.n + i];
    }
    buf[lid.x] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            buf[lid.x] += buf[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        out[c * p.count + wid.x] = buf[0];
    }
}
