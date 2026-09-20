/* HIMS Progressive Web App - Service Worker (R-15: PHI-safe caching policy) */
'use strict';

const CACHE_NAME = 'hims-pwa-v2';

/**
 * Only pre-cache these known-safe static assets.
 * All URLs must start with /static/ or be from trusted CDNs.
 */
const STATIC_ASSETS = [
  '/static/offline.html',
  '/static/manifest.json',
  '/static/favicon.ico',
];

/**
 * URL prefixes and patterns that MUST NEVER be cached by this service worker.
 * These represent PHI endpoints, API routes, billing, lab, imaging downloads, etc.
 */
const NEVER_CACHE_PATTERNS = [
  '/api/',
  '/oauth/',
  '/auth/',
  '/imaging/download',
  '/lab/download',
  '/billing/',
  '/records/',
  '/medicine/',
  '/pharmacy/',
  '/nursing/',
  '/theatre/',
  '/icu/',
  '/renal/',
  '/oncology/',
  '/compliance/',
  '/fhir/',
  '/patient/',
  '/reports/',
  '/export/',
  '/backup',
  '/admin/',
  '/sso/',
];

/**
 * Returns true if the given URL should never be cached (PHI or dynamic).
 */
function isNeverCacheUrl(url) {
  try {
    const parsed = new URL(url);
    const path = parsed.pathname;

    // Never cache non-GET or cross-origin requests to non-CDN sources
    for (const pattern of NEVER_CACHE_PATTERNS) {
      if (path.startsWith(pattern)) {
        return true;
      }
    }

    // Only cache /static/* or trusted CDN prefixes
    if (!path.startsWith('/static/')) {
      return true;
    }

    return false;
  } catch {
    return true;
  }
}

// Install Event - Pre-cache essential static assets & offline page
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[HIMS SW] Pre-caching approved static assets');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Activate Event - Clean up stale caches from previous versions
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            console.log('[HIMS SW] Purging old cache:', cache);
            return caches.delete(cache);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event - Strict PHI-safe caching policy
self.addEventListener('fetch', (event) => {
  const request = event.request;

  // Only intercept GET requests over HTTP(S)
  if (request.method !== 'GET' || !request.url.startsWith('http')) {
    return;
  }

  // HTML page navigations -> Network-first (never serve stale clinical pages)
  // This ensures patients and clinicians always see current data
  if (request.mode === 'navigate' || (request.headers.get('accept') || '').includes('text/html')) {
    event.respondWith(
      fetch(request).catch(() => {
        console.log('[HIMS SW] Network offline. Serving offline fallback page.');
        return caches.match('/static/offline.html');
      })
    );
    return;
  }

  // PHI and API endpoints -> ALWAYS go to network, NEVER cache
  if (isNeverCacheUrl(request.url)) {
    event.respondWith(
      fetch(request).catch(() => {
        return new Response('', { status: 503, statusText: 'Service Unavailable' });
      })
    );
    return;
  }

  // Static assets under /static/* -> Cache-first, update in background
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        // Background network update for freshness
        const networkFetch = fetch(request).then((networkResponse) => {
          if (
            networkResponse &&
            networkResponse.status === 200 &&
            networkResponse.type === 'basic'
          ) {
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(request, networkResponse.clone());
            });
          }
          return networkResponse;
        }).catch(() => null);
        // Return cached immediately but silently update
        return cachedResponse;
      }

      // Not in cache yet: fetch from network and cache if safe
      return fetch(request).then((networkResponse) => {
        if (
          networkResponse &&
          networkResponse.status === 200 &&
          networkResponse.type === 'basic' &&
          !isNeverCacheUrl(request.url)
        ) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(request, responseToCache);
          });
        }
        return networkResponse;
      }).catch(() => {
        return new Response('', { status: 503, statusText: 'Service Unavailable' });
      });
    })
  );
});
