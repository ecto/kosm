// Camera-relative spacecraft (metres), geocentric Earth (kilometres).
// Analytic intersections; single-scattering optical-depth integration.
struct Uniforms{eye:vec4f,forward:vec4f,right:vec4f,up:vec4f,orbit:vec4f,sun:vec4f,axis_x:vec4f,axis_y:vec4f,axis_z:vec4f,bus:vec4f,panels:vec4f,settings:vec4f}
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var earth:texture_2d<f32>;
@group(0) @binding(2) var tex_sampler:sampler;
@group(0) @binding(3) var surface_tiles:texture_2d_array<f32>;
@group(0) @binding(4) var<storage,read> tile_map:array<i32>;
@group(0) @binding(5) var cloud_map:texture_2d<f32>;
@group(0) @binding(6) var solar_lut:texture_2d<f32>;
@group(0) @binding(7) var erosion:texture_3d<f32>;
@group(0) @binding(8) var erosion_sampler:sampler;
struct VertexOut{@builtin(position) p:vec4f,@location(0) uv:vec2f}
@vertex fn vs(@builtin(vertex_index) index:u32)->VertexOut{let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));var o:VertexOut;o.p=vec4f(p[index],0.,1.);o.uv=p[index];return o;}
const PI=3.14159265;
fn sphere(o:vec3f,d:vec3f,r:f32)->vec2f{let b=dot(o,d);let h=b*b-dot(o,o)+r*r;if h<0.{return vec2f(1e20,-1e20);}let s=sqrt(h);return vec2f(-b-s,-b+s);}
fn hash(p:vec3f)->f32{return fract(sin(dot(p,vec3f(127.1,311.7,74.7)))*43758.5453);}
fn noise(p:vec3f)->f32{let i=floor(p);let f=fract(p);let a=f*f*(3.-2.*f);return mix(mix(mix(hash(i),hash(i+vec3f(1,0,0)),a.x),mix(hash(i+vec3f(0,1,0)),hash(i+vec3f(1,1,0)),a.x),a.y),mix(mix(hash(i+vec3f(0,0,1)),hash(i+vec3f(1,0,1)),a.x),mix(hash(i+vec3f(0,1,1)),hash(i+vec3f(1,1,1)),a.x),a.y),a.z);}
fn fbm(p0:vec3f)->f32{var p=p0;var f=0.;var a=0.5;for(var i=0;i<5;i++){f+=a*noise(p);p=p*2.03+vec3f(12.1,4.7,7.3);a*=0.5;}return f;}
fn rotate_earth(p:vec3f)->vec3f{let c=cos(u.sun.w);let s=sin(u.sun.w);return vec3f(c*p.x+s*p.y,-s*p.x+c*p.y,p.z);}
fn uv_map(n:vec3f)->vec2f{return vec2f(atan2(n.y,n.x)/(2.*PI)+0.5,0.5-asin(clamp(n.z,-1.,1.))/PI);}
fn clouds(n:vec3f)->f32{return textureSampleLevel(cloud_map,tex_sampler,uv_map(n),0.).r;}
fn solar_visibility(p:vec3f)->f32{let h=sphere(p,u.sun.xyz,u.orbit.w);return select(1.,0.,h.x>0.&&h.y>0.);}
fn solar_transmission(p:vec3f)->vec3f{
 let altitude=clamp(length(p)-u.orbit.w,0.,100.);let mu=dot(normalize(p),u.sun.xyz);
 if solar_visibility(p)<0.5{return vec3f(0.);}
 let xy=vec2f(mu*0.5+0.5,sqrt(altitude/100.))*vec2f(511.,127.);let lo=vec2i(floor(xy));let hi=min(lo+vec2i(1),vec2i(511,127));let f=fract(xy);
 return mix(mix(textureLoad(solar_lut,lo,0).rgb,textureLoad(solar_lut,vec2i(hi.x,lo.y),0).rgb,f.x),mix(textureLoad(solar_lut,vec2i(lo.x,hi.y),0).rgb,textureLoad(solar_lut,hi,0).rgb,f.x),f.y);
}
fn surface_data(uv0:vec2f,t:f32,incidence:f32)->vec4f{
 let uv=vec2f(fract(uv0.x),clamp(uv0.y,0.000001,0.999999));let grid=uv*vec2f(20.,10.);let cell=vec2i(floor(grid));let layer=tile_map[cell.y*20+cell.x];
 let footprint=max(t*0.0008/max(incidence,0.08)/1.956,1.);let lod=log2(footprint);
 if layer>=0{return textureSampleLevel(surface_tiles,tex_sampler,(fract(grid)*1024.+2.)/1028.,layer,lod);}
 return textureSampleLevel(earth,tex_sampler,uv,max(lod-3.32,0.));
}
fn cloud_density(p:vec3f)->f32{
 let h=length(p)-u.orbit.w;if h<1.2||h>11.{return 0.;}
 let q=rotate_earth(p);let coverage=clouds(normalize(q));
 let low=smoothstep(1.2,1.8,h)*(1.-smoothstep(2.8,4.3,h));
 let tower=smoothstep(2.4,3.4,h)*(1.-smoothstep(5.+coverage*4.,10.5,h))*smoothstep(0.58,0.88,coverage);
 let detail=textureSampleLevel(erosion,erosion_sampler,fract(q*0.018),0.).r+0.35*textureSampleLevel(erosion,erosion_sampler,fract(q*0.067),0.).r;
 let body=max(coverage-0.16-(1.-detail)*0.22,0.);
 return body*(low*0.8+tower*0.65);
}
fn cloud_shadow(p:vec3f)->f32{
 let outer=sphere(p,u.sun.xyz,u.orbit.w+11.);let inner=sphere(p,u.sun.xyz,u.orbit.w+1.2);let start=max(inner.y,0.);let end=max(outer.y,start);if end<=start{return 1.;}
 let ds=min((end-start)/8.,12.);var tau=0.;for(var i=0;i<8;i++){tau+=cloud_density(p+u.sun.xyz*(start+(f32(i)+0.5)*ds))*ds;}return exp(-tau*1.3);
}
fn space_sky(d:vec3f)->vec3f{let cell=floor(d*1100.);let rand=hash(cell);let point=fract(d*1100.);let spot=pow(max(0.,1.-length(point-vec3f(0.5))*2.),10.);var c=vec3f(0.00008,0.00012,0.0002);if rand>0.9985{c+=mix(vec3f(0.6,0.75,1.),vec3f(1.,0.78,0.5),hash(cell+3.))*spot*0.3;}let angle=acos(clamp(dot(d,u.sun.xyz),-1.,1.));c+=vec3f(1.,0.94,0.81)*smoothstep(0.0048,0.00455,angle)*40.;return c;}
fn earth_surface(o:vec3f,d:vec3f,t:f32)->vec3f{
 let p=o+d*t;let n=normalize(p);let data=surface_data(uv_map(rotate_earth(n)),t,max(dot(n,-d),0.));let ndl=max(dot(n,u.sun.xyz),0.);let nv=max(dot(n,-d),0.001);
 let halfv=normalize(u.sun.xyz-d);let nh=max(dot(n,halfv),0.);let vh=max(dot(-d,halfv),0.);
 // Wind-dependent microfacet slope variance, 5 m/s reference wind.
 let variance=0.003+0.00512*5.;let nh2=max(nh*nh,0.0001);let distribution=exp((nh2-1.)/(variance*nh2))/(PI*variance*nh2*nh2);
 let fresnel=0.0204+0.9796*pow(1.-vh,5.);let spec=distribution*fresnel/(4.*nv*max(ndl,0.001));
 let sunlight=solar_transmission(p+n*0.01)*cloud_shadow(p)*4.5;
 return (data.rgb/PI+vec3f(spec*data.a))*ndl*sunlight+data.rgb*0.006;
}
fn atmosphere(o:vec3f,d:vec3f,limit:f32,background:vec3f)->vec3f{
 let hit=sphere(o,d,u.orbit.w+100.);let start=max(hit.x,0.);let end=min(hit.y,limit);if end<=start{return background;}
 let br=vec3f(0.005802,0.013558,0.0331);let bm=vec3f(0.003996);let bo=vec3f(0.000650,0.001881,0.000085);
 let mu=dot(d,u.sun.xyz);let pr=3./(16.*PI)*(1.+mu*mu);let g=0.76;let pm=(1.-g*g)/(4.*PI*pow(1.+g*g-2.*g*mu,1.5));
 let ds=(end-start)/32.;var transmission=vec3f(1.);var scattered=vec3f(0.);
 for(var i=0;i<32;i++){
 let p=o+d*(start+(f32(i)+0.5)*ds);let h=max(length(p)-u.orbit.w,0.);let density=exp(-vec2f(h)/vec2f(8.,1.2));let ozone=max(1.-abs(h-25.)/15.,0.);
 let extinction=br*density.x+vec3f(0.00444)*density.y+bo*ozone;let step_t=exp(-extinction*ds);
 let direct=solar_transmission(p)*(br*density.x*pr+bm*density.y*pm);
 // Low-order diffuse skylight closure, deliberately separate from direct sunlight.
 let sky=vec3f(0.015,0.025,0.045)*max(dot(normalize(p),u.sun.xyz)+0.18,0.)*(br*density.x+bm*density.y);
 scattered+=transmission*(direct+sky)*4.5*(1.-step_t)/max(extinction,vec3f(1e-8));transmission*=step_t;
 }return background*transmission+scattered;
}
fn cloud_volume(o:vec3f,d:vec3f,limit:f32,background:vec3f)->vec3f{
 let outer=sphere(o,d,u.orbit.w+11.);let inner=sphere(o,d,u.orbit.w+1.2);let start=max(outer.x,0.);var end=min(outer.y,limit);if inner.x>start{end=min(end,inner.x);}if end<=start{return atmosphere(o,d,limit,background);}
 let ds=(end-start)/40.;var tr=1.;var light=vec3f(0.);var weighted=0.;var weight=0.;
 let mu=dot(d,u.sun.xyz);let phase=0.7*(1.-0.65*0.65)/pow(1.+0.65*0.65-1.3*mu,1.5)+0.3*(1.-0.2*0.2)/pow(1.+0.2*0.2+0.4*mu,1.5);
 for(var i=0;i<40;i++){
 let t=start+(f32(i)+0.5)*ds;let p=o+d*t;let density=cloud_density(p);if density<0.001{continue;}
 let absorb=1.-exp(-density*1.3*ds);var depth=0.;for(var j=0;j<6;j++){let step_s=0.6+f32(j)*0.8;depth+=cloud_density(p+u.sun.xyz*step_s)*0.8;}
 let sun=solar_transmission(p);let direct=exp(-depth*1.3)*phase*0.25;
 let multiple=0.14*exp(-depth*0.25);let ambient=vec3f(0.025,0.04,0.065)*max(dot(normalize(p),u.sun.xyz)+0.2,0.);
 light+=tr*absorb*(sun*(direct+multiple)+ambient)*4.5;weighted+=t*tr*absorb;weight+=tr*absorb;tr*=1.-absorb;if tr<0.005{break;}
 }
 if weight<0.001{return atmosphere(o,d,limit,background);}
 let depth=weighted/weight;let behind=atmosphere(o+d*depth,d,max(limit-depth,0.),background);
 return atmosphere(o,d,depth,light+behind*tr);
}
// Box hit: distance, outward face normal. Local craft coordinates preserve millimetre details.
struct Hit{t:f32,n:vec3f,mat:f32,p:vec3f}
fn box_hit(o:vec3f,d:vec3f,center:vec3f,b:vec3f,mat:f32)->Hit{let q=o-center;let inv=1./select(vec3f(1e-8),d,abs(d)>vec3f(1e-8));let a=(-b-q)*inv;let z=(b-q)*inv;let near=min(a,z);let far=max(a,z);let t=max(max(near.x,near.y),near.z);let end=min(min(far.x,far.y),far.z);var h:Hit;h.t=1e20;h.n=vec3f(0);h.mat=mat;h.p=vec3f(0);if t>0.&&t<end{h.t=t;h.n=-sign(d)*step(vec3f(t-0.0001),near);h.p=o+d*t;}return h;}
fn choose(a:Hit,b:Hit)->Hit{if b.t<a.t{return b;}return a;}
fn cylinder_hit(o:vec3f,d:vec3f,c:vec3f,r:f32,half_h:f32,mat:f32)->Hit {
 var hit:Hit;hit.t=1e20;hit.n=vec3f(0);hit.mat=mat;hit.p=vec3f(0);let q=o-c;let a=dot(d.xy,d.xy);let b=dot(q.xy,d.xy);let disc=b*b-a*(dot(q.xy,q.xy)-r*r);
 if disc>0.&&a>1e-8 {let t=(-b-sqrt(disc))/a;let p=q+d*t;if t>0.&&abs(p.z)<half_h {hit.t=t;hit.n=normalize(vec3f(p.xy,0.));hit.p=o+d*t;}}
 for(var side=-1;side<=1;side+=2){if abs(d.z)>1e-7{let t=(f32(side)*half_h-q.z)/d.z;let p=q+d*t;if t>0.&&t<hit.t&&dot(p.xy,p.xy)<r*r{hit.t=t;hit.n=vec3f(0,0,f32(side));hit.p=o+d*t;}}}return hit;
}
fn dish_hit(o:vec3f,d:vec3f)->Hit {
 let center=vec3f(-u.bus.x-0.4,0.,u.bus.z+0.1);let q=o-center;let k=0.42;let a=k*dot(d.xy,d.xy);let b=2.*k*dot(q.xy,d.xy)-d.z;let c=k*dot(q.xy,q.xy)-q.z;let disc=b*b-4.*a*c;
 var h:Hit;h.t=1e20;h.n=vec3f(0);h.mat=4.;h.p=vec3f(0);
 if disc>0.&&a>1e-7{for(var side=-1;side<=1;side+=2){let t=(-b+f32(side)*sqrt(disc))/(2.*a);let p=q+d*t;if t>0.&&t<h.t&&dot(p.xy,p.xy)<0.85*0.85{h.t=t;h.n=normalize(vec3f(-2.*k*p.xy,1.));if dot(h.n,d)>0.{h.n=-h.n;}h.p=o+d*t;}}}return h;
}
fn craft(o:vec3f,d:vec3f)->Hit{var h=box_hit(o,d,vec3f(0),u.bus.xyz,1.);let halfspan=u.panels.x*0.5;let chord=u.panels.y*0.5;
 // Six independently framed solar blankets along an articulated spar.
 h=choose(h,box_hit(o,d,vec3f(0),vec3f(0.12,halfspan,0.12),3.));
 for(var side=-1;side<=1;side+=2){for(var j=0;j<3;j++){let bay=(halfspan-u.bus.y-0.5)/3.;let y=f32(side)*(u.bus.y+0.5+(f32(j)+0.5)*bay);h=choose(h,box_hit(o,d,vec3f(0,y,0),vec3f(chord,bay*0.5-0.08,u.panels.z*0.5),2.));h=choose(h,box_hit(o,d,vec3f(0,y,0),vec3f(chord+0.045,bay*0.5-0.035,0.026),3.));}}
 h=choose(h,cylinder_hit(o,d,vec3f(0,0,u.bus.z+0.7),0.72,0.7,4.));h=choose(h,cylinder_hit(o,d,vec3f(0,0,u.bus.z+1.415),0.59,0.015,6.));h=choose(h,dish_hit(o,d));h=choose(h,box_hit(o,d,vec3f(u.bus.x+0.12,0,0),vec3f(0.12,0.91,1.0),5.));
 h=choose(h,box_hit(o,d,vec3f(0,0,u.bus.z+1.6),vec3f(0.035,0.035,0.7),3.));
 for(var i=0;i<4;i++){let x=select(-1.,1.,(i%2)==0)*u.bus.x;let y=select(-1.,1.,i<2)*u.bus.y;h=choose(h,box_hit(o,d,vec3f(x,y,-u.bus.z),vec3f(0.14,0.14,0.26),3.));}return h;}
fn craft_shade(o:vec3f,d:vec3f)->vec4f{let rotation=mat3x3f(u.axis_x.xyz,u.axis_y.xyz,u.axis_z.xyz);let inverse=transpose(rotation);let local_o=inverse*o;let local_d=inverse*d;let h=craft(local_o,local_d);if h.t>1e19{return vec4f(0.);}
 var n=normalize(h.n);let light=inverse*u.sun.xyz;let view=-local_d;let ndl=max(dot(n,light),0.);let halfv=normalize(light+view);var base=vec3f(0.64,0.37,0.07);var roughness=0.45;var metal=0.9;
 if h.mat==1.{let foil=0.88+0.12*sin(h.p.x*61.+sin(h.p.z*34.)*4.)*sin(h.p.y*53.+h.p.z*41.);base*=foil;n=normalize(n+vec3f(sin(h.p.y*47.+h.p.z*33.),sin(h.p.z*57.+h.p.x*27.),sin(h.p.x*51.+h.p.y*31.))*0.045);}
 if h.mat==2.{base=vec3f(0.014,0.028,0.085);roughness=0.16;metal=0.55;let cells=fract(h.p.xy*vec2f(9.,14.));let grid=1.-smoothstep(0.015,0.065,min(cells.x,cells.y));base=mix(base,vec3f(0.08,0.13,0.21),grid);let bands=0.75+0.25*sin(floor(h.p.x*9.)*1.7+floor(h.p.y*14.)*2.);base*=bands;}
 if h.mat==3.{base=vec3f(0.35,0.4,0.43);roughness=0.24;}
 if h.mat==4.{base=vec3f(0.65,0.7,0.72);roughness=0.5;metal=0.1;}
 if h.mat==5.{base=vec3f(0.7,0.74,0.75);roughness=0.65;metal=0.1;let slat=step(0.12,fract(h.p.z*17.));base*=0.75+0.25*slat;}
 if h.mat==6.{base=vec3f(0.008,0.015,0.025);roughness=0.09;metal=0.6;}let a=roughness*roughness;let nh=max(dot(n,halfv),0.);let nv=max(dot(n,view),0.001);let denom=nh*nh*(a*a-1.)+1.;let distribution=a*a/(PI*denom*denom);let f0=mix(vec3f(0.04),base,metal);let fresnel=f0+(1.-f0)*pow(1.-max(dot(halfv,view),0.),5.);let k=(roughness+1.)*(roughness+1.)/8.;let visibility=nv/(nv*(1.-k)+k)*ndl/(ndl*(1.-k)+k);let spec=distribution*fresnel*visibility/(4.*nv*max(ndl,0.001));let shadow=craft(h.p+n*0.005,light);let illuminated=select(0.,1.,shadow.t>1e19)*solar_visibility(u.orbit.xyz);let diffuse=(1.-fresnel)*(1.-metal)*base/PI;let earthlight=max(dot(rotation*n,-normalize(u.orbit.xyz)),0.);let reflected=rotation*reflect(local_d,n);let eh=sphere(u.orbit.xyz,reflected,u.orbit.w);var environment=vec3f(0.);if eh.x>0.&&eh.x<1e19{environment=earth_surface(u.orbit.xyz,reflected,eh.x)*f0*0.45;}let col=environment+(diffuse+spec)*ndl*4.5*illuminated+base*(vec3f(0.025,0.05,0.1)*earthlight+vec3f(0.006));return vec4f(col,1.);}
fn aces(x:vec3f)->vec3f{return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),vec3f(0.),vec3f(1.));}
@fragment fn fs(v:VertexOut)->@location(0) vec4f{let d=normalize(u.forward.xyz+u.right.xyz*v.uv.x*u.eye.w*u.forward.w+u.up.xyz*v.uv.y*u.forward.w);let o=u.orbit.xyz;let hit=sphere(o,d,u.orbit.w);var col=space_sky(d);var distance=1e20;if hit.x>0.&&hit.x<1e19{distance=hit.x;col=earth_surface(o,d,hit.x);}col=cloud_volume(o,d,distance,col);if u.settings.x<0.5{let ship=craft_shade(u.eye.xyz,d);col=mix(col,ship.rgb,ship.a);}col=aces(col*u.right.w);return vec4f(pow(col,vec3f(1./2.2)),1.);}
