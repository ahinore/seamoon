import * as THREE from 'three';

/**
 * Planet material: single color + Lambert term against a fixed sun direction.
 * Deliberately a custom ShaderMaterial so Phases 3-5 can replace the shading
 * without touching the render pipeline. Depth handling is reversed-Z: the
 * renderer clears depth to 0 and tests GreaterEqual, and the camera's
 * projection matrix flips the z axis, so shaders need no depth output work.
 */
export function makePlanetMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSunDir: { value: new THREE.Vector3(1, 0.3, 0.35).normalize() },
      uBase: { value: new THREE.Color(0x1f5fa8) },
      uWire: { value: 0 },
    },
    vertexShader: /* glsl */ `
      #include <common>
      attribute vec3 center;
      varying vec3 vN;
      varying vec3 vC;
      void main() {
        vN = normalize(mat3(modelMatrix) * normal);
        vC = center;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSunDir;
      uniform vec3 uBase;
      uniform float uWire;
      varying vec3 vN;
      varying vec3 vC;
      void main() {
        float ndl = clamp(dot(normalize(vN), uSunDir), 0.0, 1.0);
        vec3 col = uBase * (0.05 + 0.95 * ndl);
        if (uWire > 0.5) col = mix(col, vec3(1.0), 0.45);
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

/** Simple background star points (placeholder until Phase 9). */
export function makeStars(count = 2500, radius = 6e8): THREE.Points {
  const pos = new Float32Array(count * 3);
  const mag = new Float32Array(count);
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    // uniform direction sampling
    const z = Math.random() * 2 - 1;
    const phi = Math.random() * Math.PI * 2;
    const r = Math.sqrt(1 - z * z);
    v.set(r * Math.cos(phi), z, r * Math.sin(phi)).multiplyScalar(radius);
    pos[i * 3] = v.x;
    pos[i * 3 + 1] = v.y;
    pos[i * 3 + 2] = v.z;
    mag[i] = 0.35 + Math.random() * 0.65;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aMag', new THREE.BufferAttribute(mag, 1));
  const m = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 1.6 } },
    vertexShader: /* glsl */ `
      #include <common>
      uniform float uSize;
      attribute float aMag;
      varying float vMag;
      void main() {
        vMag = aMag;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uSize;
      }
    `,
    fragmentShader: /* glsl */ `
      varying float vMag;
      void main() {
        gl_FragColor = vec4(vec3(0.8 + 0.2 * vMag), 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  return pts;
}
