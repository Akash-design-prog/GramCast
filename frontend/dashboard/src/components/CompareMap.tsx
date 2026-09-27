import { useEffect, useRef, useState } from "react";
import { Map as MLMap, GeoJSONSource } from "maplibre-gl";
import type { StyleSpecification, FilterSpecification, MapMouseEvent } from "maplibre-gl";
import type { ForecastMapResponse, VillageProperties } from "../lib/types";
import { rainfallFillExpression } from "../lib/rainfallStyle";
import { boundsOf } from "../lib/geo";

const SOURCE_ID = "villages";
const LAYER_ID = "villages-fill";
const OUTLINE_ID = "villages-outline";
const SELECTED_ID = "villages-selected";

// No basemap tiles: the village mosaic is the whole point of this view, and a busy street/satellite
// layer would fight the calm palette instead of supporting it. The container's own CSS background
// (theme-aware) shows through instead of a MapLibre background layer.
const BLANK_STYLE: StyleSpecification = { version: 8, sources: {}, layers: [] };

interface Props {
  data: ForecastMapResponse | null;
  onVillageClick: (props: VillageProperties, featureId: number, lngLat: { lat: number; lng: number }) => void;
  selectedFeatureId: number | null;
  // A new object each time (even for the same coordinates) so re-selecting the same search result still
  // re-triggers the fly-to - see the App.tsx handler that builds this.
  flyTarget: { lat: number; lon: number; key: number } | null;
}

function buildLayers(map: MLMap, colorField: string, outline: { color: string; width: number }) {
  map.addSource(SOURCE_ID, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({
    id: LAYER_ID,
    type: "fill",
    source: SOURCE_ID,
    paint: { "fill-color": rainfallFillExpression(colorField), "fill-opacity": 0.92 },
  });
  map.addLayer({
    id: OUTLINE_ID,
    type: "line",
    source: SOURCE_ID,
    paint: { "line-color": outline.color, "line-width": outline.width },
  });
  map.addLayer({
    id: SELECTED_ID,
    type: "line",
    source: SOURCE_ID,
    paint: { "line-color": "#26241f", "line-width": 2.5 },
    filter: ["==", ["id"], -1],
  });
}

const VILLAGE_OUTLINE = { color: "rgba(0,0,0,0.12)", width: 0.4 };
// Bolder, more opaque lines than the village outline - a real 0.25deg grid should read as a handful of
// clean squares, not blend into the same faint mosaic style used for 2,003 village boundaries.
const GRID_OUTLINE = { color: "rgba(0,0,0,0.35)", width: 1.2 };

/** One MapLibre instance with no basemap, just a choropleth (villages or grid cells) colored by
 * `colorField`. */
function useMapPane(containerRef: React.RefObject<HTMLDivElement | null>, colorField: string, outline: { color: string; width: number }) {
  const mapRef = useRef<MLMap | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const map = new MLMap({
      container: containerRef.current,
      style: BLANK_STYLE,
      center: [73.85, 18.52],
      zoom: 7.6,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
    });
    map.on("load", () => buildLayers(map, colorField, outline));
    mapRef.current = map;

    // Two related but distinct issues, both fixed here:
    // 1. MapLibre sizes its internal camera to the container's dimensions at construction time and
    //    never re-checks - if the container isn't at its final size yet (flex/grid layout still
    //    settling, web fonts still loading) the map renders nothing even though data is fine.
    //    ResizeObserver catches this whenever the container's real size actually changes.
    // 2. If the tab is hidden (backgrounded) at the exact moment the map is constructed, MapLibre's
    //    internal render loop (driven by requestAnimationFrame, which browsers suspend for hidden
    //    tabs) never gets its first tick and gets stuck forever - 'load' never fires even after the
    //    tab later becomes visible, because nothing re-kicks it. Confirmed directly: a map stuck this
    //    way stayed blank indefinitely until an explicit resize() forced a repaint. A 'visibilitychange'
    //    listener that nudges resize() whenever the page becomes visible recovers from this reliably -
    //    this is the scenario a real user hits by opening the dashboard in a background tab and
    //    switching to it later, not just a test-automation artifact.
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(containerRef.current);
    const onVisible = () => {
      if (document.visibilityState === "visible") map.resize();
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      observer.disconnect();
      map.remove();
    };
    // colorField/outline are fixed per pane for the lifetime of the component - no need to react to them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerRef]);

  return mapRef;
}

export default function CompareMap({ data, onVillageClick, selectedFeatureId, flyTarget }: Props) {
  const blockContainer = useRef<HTMLDivElement>(null);
  const panchayatContainer = useRef<HTMLDivElement>(null);
  const blockMap = useMapPane(blockContainer, "block_mm", GRID_OUTLINE);
  const panchayatMap = useMapPane(panchayatContainer, "p50_mm", VILLAGE_OUTLINE);

  const stageRef = useRef<HTMLDivElement>(null);
  const [slider, setSlider] = useState(55);
  const dragging = useRef(false);
  const hasFitBounds = useRef(false);

  // Push new data into both sources once loaded, and once per dataset, fit the camera to the
  // district's real extent instead of a hardcoded zoom guess.
  useEffect(() => {
    if (!data) return;
    // The block pane gets the real 0.25deg grid rectangles (grid_cells), not the village polygons -
    // same block_mm data, but rendered as actual grid squares instead of village-shaped patches.
    const panes: [MLMap | null, GeoJSON.FeatureCollection][] = [
      [blockMap.current, data.grid_cells as unknown as GeoJSON.FeatureCollection],
      [panchayatMap.current, data as unknown as GeoJSON.FeatureCollection],
    ];
    for (const [map, collection] of panes) {
      if (!map) continue;
      const src = map.getSource(SOURCE_ID) as GeoJSONSource | undefined;
      if (src) src.setData(collection);
      else map.once("load", () => (map.getSource(SOURCE_ID) as GeoJSONSource)?.setData(collection));
    }
    if (!hasFitBounds.current) {
      const [west, south, east, north] = boundsOf(data);
      const target = blockMap.current;
      const apply = (map: MLMap) => map.fitBounds([[west, south], [east, north]], { padding: 24, duration: 0 });
      if (target) {
        if (target.loaded()) apply(target);
        else target.once("load", () => apply(target));
      }
      hasFitBounds.current = true;
    }
  }, [data, blockMap, panchayatMap]);

  // Search result selected: fly both cameras to it directly (rather than relying on the move-sync
  // effect below to propagate one pane's animated flyTo to the other, which would fight the animation).
  useEffect(() => {
    if (!flyTarget) return;
    for (const map of [blockMap.current, panchayatMap.current]) {
      if (!map) continue;
      map.flyTo({ center: [flyTarget.lon, flyTarget.lat], zoom: Math.max(map.getZoom(), 10.5), duration: 800 });
    }
  }, [flyTarget, blockMap, panchayatMap]);

  // Camera sync: forward the block map's view to the panchayat map (and back), guarded against
  // feedback loops with a "syncing" flag.
  useEffect(() => {
    const a = blockMap.current;
    const b = panchayatMap.current;
    if (!a || !b) return;
    let syncing = false;
    const sync = (from: MLMap, to: MLMap) => () => {
      if (syncing) return;
      syncing = true;
      to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
      syncing = false;
    };
    const aToB = sync(a, b);
    const bToA = sync(b, a);
    a.on("move", aToB);
    b.on("move", bToA);
    return () => {
      a.off("move", aToB);
      b.off("move", bToA);
    };
  }, [blockMap, panchayatMap]);

  // Click handling on the panchayat (fine) layer only - that's the layer with per-village detail.
  useEffect(() => {
    const map = panchayatMap.current;
    if (!map) return;
    const handler = (e: MapMouseEvent) => {
      const features = map.queryRenderedFeatures(e.point, { layers: [LAYER_ID] });
      const feature = features[0];
      if (feature && feature.id !== undefined) {
        onVillageClick(feature.properties as VillageProperties, feature.id as number, { lat: e.lngLat.lat, lng: e.lngLat.lng });
      }
    };
    map.on("click", LAYER_ID, handler);
    const cursorIn = () => (map.getCanvas().style.cursor = "pointer");
    const cursorOut = () => (map.getCanvas().style.cursor = "");
    map.on("mouseenter", LAYER_ID, cursorIn);
    map.on("mouseleave", LAYER_ID, cursorOut);
    return () => {
      map.off("click", LAYER_ID, handler);
      map.off("mouseenter", LAYER_ID, cursorIn);
      map.off("mouseleave", LAYER_ID, cursorOut);
    };
  }, [panchayatMap, onVillageClick]);

  // Highlight the selected village - panchayat pane only. Filtering by the feature's real id (not by
  // name - 383/2003 villages share a name with at least one other village elsewhere in the district,
  // e.g. "Shindewadi" appears in 6 different talukas, so a name-based filter lit up every same-named
  // village on the map instead of just the one clicked). The block pane's source is grid_cells now, a
  // completely different feature set (0-69) whose ids can coincidentally collide with a village id in
  // that same range, so it never gets a selection filter applied.
  useEffect(() => {
    const filter: FilterSpecification = ["==", ["id"], selectedFeatureId ?? -1];
    const map = panchayatMap.current;
    if (map?.getLayer(SELECTED_ID)) map.setFilter(SELECTED_ID, filter);
  }, [selectedFeatureId, panchayatMap]);

  function pctFromClientX(clientX: number): number {
    const rect = stageRef.current!.getBoundingClientRect();
    return Math.max(6, Math.min(94, ((clientX - rect.left) / rect.width) * 100));
  }

  return (
    <div className="compare-map">
      <div className="compare-map__labels">
        <span className="pill">25km block forecast</span>
        <span className="pill">5km panchayat forecast</span>
      </div>
      <div
        ref={stageRef}
        className="compare-map__stage"
        onPointerMove={(e) => dragging.current && setSlider(pctFromClientX(e.clientX))}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => (dragging.current = false)}
      >
        <div ref={blockContainer} className="compare-map__pane" />
        <div ref={panchayatContainer} className="compare-map__pane" style={{ clipPath: `inset(0 0 0 ${slider}%)` }} />
        <div
          className="compare-map__handle"
          style={{ left: `${slider}%` }}
          onPointerDown={(e) => {
            dragging.current = true;
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
          }}
        >
          <div className="compare-map__knob">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M8 7l-5 5 5 5M16 7l5 5-5 5" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
