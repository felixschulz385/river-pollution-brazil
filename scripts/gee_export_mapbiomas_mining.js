// MapBiomas Collection 11 mining substances -> Google Drive.
//
// Paste into the Earth Engine Code Editor (code.earthengine.google.com) and Run.
// The Console prints the asset's bands, properties and the class values found
// around Itaituba (Tapajós). Two export tasks then appear in the Tasks tab;
// click Run on each:
//   1. mining_substances_area_by_class_year (CSV): national area (km2) by class
//      value and band (year), to check for the 2020 step before any transfer.
//   2. mapbiomas_mining_substances_c11 (GeoTIFF): all bands (years) for Brazil
//      on the asset's own grid, split into 12288x12288-pixel files; tiles
//      without mining are skipped.
// Files land in the Drive folder below; rsync them to transfer12 and merge
// per year on the HPC.
//
// Class codes are ABCC: A = 1 garimpo (artisanal), 2 industrial; B = 1 exposed
// soil, 2 revegetation; CC = substance (15 gold, 14 tin, 09 copper, 24 precious
// stones, 59 no substance, ...). Legend CSV: https://brasil.mapbiomas.org/wp-content/
// uploads/sites/3/2026/10/legend_code_mapbiomas_mining_brazil_collection_11_mined_substances.zip

var ASSET = 'projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_mining_substances_v1';
var DRIVE_FOLDER = 'mapbiomas_mining_c11';
// Multiple of the 256-pixel shard size; 41 uint16 bands (82 bytes/pixel) must stay
// under GEE's 16 GiB uncompressed limit per file: 12288^2 * 82 B = 12.4 GB.
var TILE_PIXELS = 12288;

var image = ee.Image(ASSET);
var brazil = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
  .filter(ee.Filter.eq('country_na', 'Brazil'))
  .geometry();

// --- 1. What is in the asset -------------------------------------------------
var bandNames = image.bandNames();
print('Band names', bandNames);
print('Properties', image.toDictionary());
var projection = image.select(0).projection().getInfo();
print('Projection (export grid)', projection);

var itaituba = ee.Geometry.Rectangle([-57.5, -7.5, -55.0, -4.0]);
var lastBand = ee.String(bandNames.get(bandNames.size().subtract(1)));
print('Class values near Itaituba, last band', lastBand,
  image.select(lastBand).reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: itaituba,
    scale: 30,
    maxPixels: 1e10
  }));

// --- 2. National area by class and year (batch, no Console time limit) --------
var area = ee.Image.pixelArea().divide(1e6);
var rows = ee.FeatureCollection(bandNames.map(function (band) {
  band = ee.String(band);
  var stats = area.addBands(image.select(band).rename('class')).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: brazil,
    scale: 30,
    maxPixels: 1e13,
    tileScale: 4
  });
  var groups = ee.List(stats.get('groups'));
  return ee.FeatureCollection(groups.map(function (group) {
    group = ee.Dictionary(group);
    return ee.Feature(null, {band: band, 'class': group.get('class'), area_km2: group.get('sum')});
  }));
})).flatten();

Export.table.toDrive({
  collection: rows,
  description: 'mining_substances_area_by_class_year',
  folder: DRIVE_FOLDER,
  fileFormat: 'CSV',
  selectors: ['band', 'class', 'area_km2']
});

// --- 3. Rasters: all years for Brazil, in tiles ------------------------------
// Class codes are four digits (e.g. 1115, 2259), so 16 bits are needed.
Export.image.toDrive({
  image: image.toUint16(),
  description: 'mapbiomas_mining_substances_c11',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'mapbiomas_mining_substances_c11',
  region: brazil.bounds(),
  crs: projection.crs,
  crsTransform: projection.transform,
  maxPixels: 1e13,
  fileDimensions: TILE_PIXELS,
  skipEmptyTiles: true,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: 0}
});
