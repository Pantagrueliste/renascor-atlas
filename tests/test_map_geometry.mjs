import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const code=readFileSync(new URL('../web/map-geometry.js',import.meta.url),'utf8');
const unwrap=runInNewContext(code+'\nunwrapLand');
const feature=(coordinates,type='Polygon')=>({type:'FeatureCollection',features:[{type:'Feature',geometry:{type,coordinates},properties:{}}]});
// Exterior crosses the date line; a hole originally at -178 must move to +182.
const outer=[[170,0],[-170,0],[-170,20],[170,20],[170,0]];
const hole=[[-178,5],[-178,10],[-175,10],[-175,5],[-178,5]];
const f=unwrap(feature([outer,hole]));
assert.deepEqual(Array.from(f.features[0].geometry.coordinates[1][0]),[182,5]);
assert.equal(f.features[0].geometry.coordinates[0][1][0],190);
// Ordinary polygons retain both their coastlines and inland-water cutouts.
const normal=feature([[[0,0],[20,0],[20,20],[0,20],[0,0]],[[5,5],[5,10],[10,10],[10,5],[5,5]]]);
const before=JSON.stringify(normal);
assert.equal(JSON.stringify(unwrap(normal)),before);
// The same correction applies inside a MultiPolygon.
assert.equal(unwrap(feature([[outer,hole]],'MultiPolygon')).features[0].geometry.coordinates[0][1][0][0],182);
console.log('3 polygon alignment checks passed');
