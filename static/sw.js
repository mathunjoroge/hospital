/* HIMS Progressive Web App - Service Worker */
const CACHE_NAME = 'hims-pwa-v1';
const STATIC_ASSETS = [
  '/static/offline.html',
  '/static/manifest.json',
  '/static/favicon.ico',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css',
  'https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css'
];

// Install Event - Pre-cache essential static assets & offline page
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[HIMS SW] Pre-caching static assets and offline page');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Activate Event - Clean up stale caches
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

// Fetch Event - Network-first for navigation/pages, Cache-first for static assets
self.addEventListener('fetch', (event) => {
  const request = event.request;

  // Ignore non-GET requests or WebSocket connections
  if (request.method !== 'GET' || !request.url.startsWith('http')) {
    return;
  }

  // HTML page navigations -> Network-first, fallback to /static/offline.html
  if (request.mode === 'navigate' || request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(
      fetch(request)
        .catch(() => {
          console.log('[HIMS SW] Network offline. Serving offline fallback page.');
          return caches.match('/static/offline.html');
        })
    );
    return;
  }

  // Static assets -> Cache-first, fallback to network
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(request, responseToCache);
          });
        }
        return networkResponse;
      }).catch(() => {
        // Return null or basic fallback if asset fetch fails offline
        return new Response('', { status: 503, statusText: 'Service Unavailable' });
      });
    })
  );
});
