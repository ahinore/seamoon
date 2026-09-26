import type { BodySurface } from './tileGeometry';
import { moonHeight, moonColor, MOON_MAX_ELEV } from './moon';

/**
 * The moon as a BodySurface (Phase 9): plugs the crater height field and
 * regolith palette into the shared cube-sphere tile pipeline.
 */
export const MOON_BODY: BodySurface = {
  height: moonHeight,
  color: moonColor,
  maxElev: MOON_MAX_ELEV,
};
