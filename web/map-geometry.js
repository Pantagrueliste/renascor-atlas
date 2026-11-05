// Keep a polygon's holes on the same longitude branch as its exterior.
// Unwrapping Eurasia can move it west by 360°; the Caspian must move with it.
function unwrapLand(fc) {
  const middleLongitude = ring => {
    let min = Infinity, max = -Infinity;
    for (const [lon] of ring) { min = Math.min(min, lon); max = Math.max(max, lon); }
    return (min + max) / 2;
  };
  for (const feature of fc.features) {
    const polygons = feature.geometry.type === 'Polygon' ? [feature.geometry.coordinates] : feature.geometry.coordinates;
    for (const polygon of polygons) {
      for (const ring of polygon) {
        let offset = 0;
        for (let i = 1; i < ring.length; i++) {
          const delta = ring[i][0] + offset - ring[i - 1][0];
          if (delta > 180) offset -= 360;
          else if (delta < -180) offset += 360;
          ring[i] = [ring[i][0] + offset, ring[i][1]];
        }
        const gap = ring[ring.length - 1][0] - ring[0][0];
        if (Math.abs(gap) > 180) {
          const pole = ring.reduce((sum, c) => sum + c[1], 0) < 0 ? -90 : 90;
          ring.splice(ring.length - 1, 0, [ring[ring.length - 1][0], pole], [ring[0][0], pole]);
          ring[ring.length - 1] = ring[0].slice();
        }
      }
      const center = middleLongitude(polygon[0]);
      for (const hole of polygon.slice(1)) {
        const shift = 360 * Math.round((center - middleLongitude(hole)) / 360);
        for (const coordinate of hole) coordinate[0] += shift;
      }
    }
  }
  return fc;
}
