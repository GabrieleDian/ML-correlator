/**
 * This file contains the logic for loading third-party analytics scripts. As of this writing,
 * we have the following vendors:
 *   Analytics:
 *      GoogleTagManager
 *      SegmentAnalytics
 *      IntercomChat
 *      ZendeskChat
 *      WebSights
 *      Datadog
 *   Application Features:
 *      Pendo
 *      Chameleon
 *      BeamerFeatureUpdates
 *
 * The way that this system works is as follows:
 *    1. Each of the above vendors has a function like `enable[VENDOR]` which is responsible for inserting scripts/doing initialization.
 *    2. In the `main` function, we have 3 code paths:
 *        1. (Dev) Analytics disabled: do nothing
 *        2. (Public) Analytics enabled, third-party analytics enabled: insert scripts for third-party analytics
 *        3. (Private) Private environment - select third-party analytics.
 *       Cases 1 and 3 are quite simple, so let's focus on #2, which is defined in `setupAnalyticsMain`.
 *    3. `setupAnalyticsMain` initializes the `window.WBAnalyticsInjector` object, which is responsible for inserting scripts
 *       based on authentication and cookie consent. It has the following affordances:
 *        * `setScriptForGroupId`: This function takes a cookie consent group ID and a callback. The callback is executed
 *          when/if the cookie consent group is enabled - after calling `initializeTrackingScripts`. This is used to insert
 *          scripts for third-party analytics. As of this writing, we have 5 groups defined in OneTrust, but only 2 are used
 *          for analytics (C0002 and C0003). See the implementation of `setupAnalyticsMain` for more details.
 *        * `initializeTrackingScripts`: This function is responsible for actually initializing the scripts. It takes a boolean
 *          `isAuthenticated` which is true if the user is authenticated. Since our tracking policy is included in ToS, we always
 *           initialize scripts if the user is authenticated. Otherwise, we use OneTrust to determine if the user has consented to
 *           to the corresponding groups.
 *    4. Importantly, the `WBAnalyticsInjector` is a global singleton. It is actually used in `consentAwareAnalyticSetup.tsx` which will
 *       call `initializeTrackingScripts` with the current authentication state. The result will be one of 3 conditions:
 *           1. User is authenticated: all scripts are initialized
 *           2. User is not authenticated, but has specified their consent: scripts corresponding to the consented groups are initialized
 *           3. User is not authenticated, and has not specified their consent: no scripts are initialized - the banner is shown and once consent
 *              is given, the scripts are initialized.
 */

window.DEBUG_ANALYTICS = false;

/// UTILS
function debugLog(...args) {
  if (window.DEBUG_ANALYTICS) {
    console.log(...args);
  }
}

function insertScript(src, attributes) {
  // Helper function to insert a script into the document.
  return new Promise((resolve, reject) => {
    // Unconditionally add a script to the document.
    var script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = reject;
    if (attributes) {
      Object.keys(attributes).forEach(key => {
        script.setAttribute(key, attributes[key]);
      });
    }
    document.body.appendChild(script);
  });
}

/// SEGMENT ANALYTICS
function enableSegmentAnalytics() {
  const analytics = window.analytics;

  // Adds the segment analytics library. This is boilerplate.
  if (!analytics.initialize) {
    analytics.SNIPPET_VERSION = '4.0.0';
    // this is segment's analytics.js library (https://github.com/segmentio/analytics.js-core) bundled with all of our integrations and settings.
    // it can be retrieved from https://cdn.segment.com/analytics.js/v1/WmJvIs1MTqNjKAeQmEyw6TvqyRI5Su2z/analytics.min.js
    // we serve it locally since cdn.segment.com is usually blocked by ad-blockers.
    let prom;
    if (
      window.CONFIG?.ENVIRONMENT_NAME === 'development' ||
      window.CONFIG?.ENVIRONMENT_NAME === 'qa'
    ) {
      prom = insertScript(
        'https://cdn.segment.com/analytics.js/v1/FDppQ6oWt2M4gWarXmIdanSQmj3BjPkl/analytics.min.js'
      ).then(analyticsCallback);
    } else {
      prom = insertScript(
        (window.CONFIG?.PUBLIC_URL ?? '') + '/sa.min.js'
      ).then(analyticsCallback);
    }

    // autotrack.js provides the trackers required below
    return prom.then(() =>
      insertScript((window.CONFIG?.PUBLIC_URL ?? '') + '/autotrack.js').then(
        () => {
          window.ga =
            window.ga ||
            function () {
              window.ga.q = window.ga.q || [];
              window.ga.q.push(arguments);
            };
          window.ga('require', 'eventTracker');
          window.ga('require', 'outboundLinkTracker');
          window.ga('require', 'urlChangeTracker');
          window.ga('require', 'cleanUrlTracker');
          window.ga('require', 'impressionTracker');
          window.ga('require', 'maxScrollTracker');
          window.ga('require', 'mediaQueryTracker');
          window.ga('require', 'outboundFormTracker');
          window.ga('require', 'pageVisibilityTracker');

          debugLog('Segment analytics initialized');
        }
      )
    );
  }
}

function enableSegmentAnalyticsLogging() {
  const analytics = window.analytics;
  Object.getOwnPropertyNames(analytics)
    .filter(p => {
      return typeof analytics[p] === 'function';
    })
    .forEach(p => {
      const f = analytics[p];
      analytics[p] = function () {
        if (p === 'page') {
          console.groupCollapsed('segment:' + p + ' ' + arguments[1]);
          console.log('category: ', arguments[0]);
          console.log('properties: ', arguments[2]);
          console.groupEnd();
        } else if (p === 'track' || p === 'group' || p === 'identify') {
          console.groupCollapsed('segment:' + p + ' ' + arguments[0]);
          console.log('properties: ', arguments[1]);
          console.groupEnd();
        }
        return f.apply(this, arguments);
      };
    });
}

function addSegmentAnalyticsMethods() {
  var analytics = (window.analytics = window.analytics || []);
  // Segment boilerplate. Add stubs for segment analytics methods so
  // they can be called even if segment is disabled.
  if (analytics.invoked) {
    console.error('Segment snippet included twice.');
  } else {
    analytics.invoked = !0;
    analytics.methods = [
      'trackSubmit',
      'trackClick',
      'trackLink',
      'trackForm',
      'pageview',
      'identify',
      'reset',
      'group',
      'track',
      'ready',
      'alias',
      'debug',
      'page',
      'once',
      'off',
      'on',
      'use',
      'init',
      'user',
      'anonymousId',
    ];
    analytics.factory = function (t) {
      return function () {
        var e = Array.prototype.slice.call(arguments);
        e.unshift(t);
        analytics.push(e);
        return analytics;
      };
    };
    for (var t = 0; t < analytics.methods.length; t++) {
      var e = analytics.methods[t];
      analytics[e] = analytics.factory(e);
    }
  }

  if (window.document.cookie.indexOf('segment_logging') != -1) {
    enableSegmentAnalyticsLogging();
  }
}

function analyticsCallback() {
  // Shutdown Fullstory script upon load, and only restart if user is logged in (util/analytics.ts).
  if (
    window.FS != null &&
    window.FS.shutdown != null &&
    !window.dontShutdownFS
  ) {
    window.FS.shutdown();
  }

  if (window.document.cookie.indexOf('segment_logging') !== -1) {
    enableSegmentAnalyticsLogging();
  }
}

/// GOOGLE Tags
function enableGoogleTagManager() {
  // Google Tag Manager
  (function (w, d, s, l, i) {
    w[l] = w[l] || [];
    w[l].push({'gtm.start': new Date().getTime(), event: 'gtm.js'});
    var f = d.getElementsByTagName(s)[0],
      j = d.createElement(s),
      dl = l !== 'dataLayer' ? '&l=' + l : '';
    j.async = true;
    j.src = 'https://www.googletagmanager.com/gtm.js?id=' + i + dl;
    f.parentNode.insertBefore(j, f);
  })(window, document, 'script', 'dataLayer', 'GTM-TCTWFHR');
  // Google Optimize
  //       (function(a,s,y,n,c,h,i,d,e){s.className+=' '+y;h.start=1*new Date;
  //   h.end=i=function(){s.className=s.className.replace(RegExp(' ?'+y),'')};
  //   (a[n]=a[n]||[]).hide=h;setTimeout(function(){i();h.end=null},c);h.timeout=c;
  // })(window,document.documentElement,'async-hide','dataLayer',4000,
  //   {'OPT-NSHDK56':true});
  //       insertCookieControlledScript("https://www.googleoptimize.com/optimize.js?id=OPT-NSHDK56", 2);
  // Google Tag Manager noscript
  const noscript = document.createElement('noscript');
  noscript.innerText =
    '<iframe src="https://www.googletagmanager.com/ns.html?id=GTM-TCTWFHR" height="0" width="0" style="display:none;visibility:hidden"></iframe>';
  document.body.appendChild(noscript);
  debugLog('Google Tag Manager initialized');
}

/// BEAMER
function enableBeamerFeatureUpdates() {
  // Beamer feature updates
  window.beamer_config = {
    product_id: 'iTpiKrhl12143',
    lazy: true,
  };
  return insertScript('https://app.getbeamer.com/js/beamer-embed.js').then(
    () => {
      debugLog('Beamer feature updates initialized');
    }
  );
}

/// WebSights
function enableWebSights() {
  return insertScript(
    'https://ws.zoominfo.com/pixel/qhXboAuxB56YVRvedMdX'
  ).then(() => {
    debugLog('WebSights initialized');
  });
}

/// Chameleon
function enableChameleon() {
  /* chameleon.io script */

  // eslint-disable-next-line no-unused-expressions
  !(function (d, w) {
    var t =
        'SqkNPmCYadc3iHUMoLcC8X2UT6Yemjzv64zMuQSY2t4zin-1PnkYf-EbS0F2wTevoZpZQX',
      c = 'chmln',
      i = d.createElement('script');
    if ((w[c] || (w[c] = {}), !w[c].root)) {
      // eslint-disable-next-line no-unused-expressions, no-sequences
      (w[c].accountToken = t),
        (w[c].location = w.location.href.toString()),
        (w[c].now = new Date()),
        (w[c].fastUrl = 'https://fast.chameleon.io/');
      var m =
        'identify alias track clear set show on off custom help _data'.split(
          ' '
        );
      for (var s = 0; s < m.length; s++) {
        // eslint-disable-next-line no-unused-expressions, no-loop-func
        !(function () {
          var t = (w[c][m[s] + '_a'] = []);
          w[c][m[s]] = function () {
            t.push(arguments);
          };
        })();
      }
      // eslint-disable-next-line no-unused-expressions, no-sequences
      (i.src = w[c].fastUrl + 'messo/' + t + '/messo.min.js'),
        (i.async = !0),
        d.head.appendChild(i);
    }
  })(document, window);

  debugLog('Chameleon initialized');
}

//Mixpanel
function enableMixpanel() {
  (function (f, b) { if (!b.__SV) { var e, g, i, h; window.mixpanel = b; b._i = []; b.init = function (e, f, c) { function g(a, d) { var b = d.split("."); 2 == b.length && ((a = a[b[0]]), (d = b[1])); a[d] = function () { a.push([d].concat(Array.prototype.slice.call(arguments, 0))); }; } var a = b; "undefined" !== typeof c ? (a = b[c] = []) : (c = "mixpanel"); a.people = a.people || []; a.toString = function (a) { var d = "mixpanel"; "mixpanel" !== c && (d += "." + c); a || (d += " (stub)"); return d; }; a.people.toString = function () { return a.toString(1) + ".people (stub)"; }; i = "disable time_event track track_pageview track_links track_forms track_with_groups add_group set_group remove_group register register_once alias unregister identify name_tag set_config reset opt_in_tracking opt_out_tracking has_opted_in_tracking has_opted_out_tracking clear_opt_in_out_tracking start_batch_senders people.set people.set_once people.unset people.increment people.append people.union people.track_charge people.clear_charges people.delete_user people.remove".split( " "); for (h = 0; h < i.length; h++) g(a, i[h]); var j = "set set_once union unset remove delete".split(" "); a.get_group = function () { function b(c) { d[c] = function () { call2_args = arguments; call2 = [c].concat(Array.prototype.slice.call(call2_args, 0)); a.push([e, call2]); }; } for ( var d = {}, e = ["get_group"].concat( Array.prototype.slice.call(arguments, 0)), c = 0; c < j.length; c++) b(j[c]); return d; }; b._i.push([e, f, c]); }; b.__SV = 1.2; e = f.createElement("script"); e.type = "text/javascript"; e.async = !0; e.src = "undefined" !== typeof MIXPANEL_CUSTOM_LIB_URL ? MIXPANEL_CUSTOM_LIB_URL : "file:" === f.location.protocol && "//cdn.mxpnl.com/libs/mixpanel-2-latest.min.js".match(/^\/\//) ? "https://cdn.mxpnl.com/libs/mixpanel-2-latest.min.js" : "//cdn.mxpnl.com/libs/mixpanel-2-latest.min.js"; g = f.getElementsByTagName("script")[0]; g.parentNode.insertBefore(e, g); } })(document, window.mixpanel || []);
  debugLog('Mixpanel initialized');
}

/// Datadog
function enableDatadog() {
  // Datadog has an option to load their library with async CDN similar
  // to other scripts in this file, however I found in testing that
  // it was very flaky in actually reporting results back to the
  // servers, so we are using the npm package and only initializing it
  // if we've set this flag.
  window.DatadogEnabled = true;

  debugLog('Datadog initialized');
}

/// Zendesk Chat
function enableZendeskChat() {
  // Don't load on kaggle.com, colab.research.google.com, and iOS/Android
  // (behavior borrowed from Intercom)
  const kaggleRegex = /kaggle\.com/;
  const colabRegex = /colab\.research\.google\.com/;
  const mobileRegex = /iPhone|iPad|Android/;
  const isKaggle = kaggleRegex.test(window.location.host);
  const isColab = colabRegex.test(window.location.host);
  const isMobile = mobileRegex.test(navigator.userAgent);
  if (isKaggle || isColab || isMobile) {
    return;
  }

  const src =
    'https://static.zdassets.com/ekr/snippet.js?key=a0a3439e-c6f9-4864-8d97-4bc7bef38531';
  // the zendesk chat script explicitly needs the 'ze-snippet' ID set on the element
  // before the script loads, otherwise it doesn't work.
  return insertScript(src, {id: 'ze-snippet'})
    .then(() => {
      const scriptEl = document.querySelector(`script[src="${src}"]`);
      if (scriptEl != null) {
        // Hid custom launcher since we want users to go through menu
        // https://github.com/wandb/core/pull/20971

        // hide default launcher and set custom launcher behavior
        const customLauncherScript = document.createElement('script');
        customLauncherScript.type = 'text/javascript';
        customLauncherScript.innerHTML = `zE('webWidget', 'hide')
      function openWidget() {
        zE('webWidget', 'show');
        zE('webWidget', 'open');
      }
      zE('webWidget:on', 'close', function() {
        zE('webWidget', 'hide');

      })`;
        // include launch behavior after zendesk chat script loads
        scriptEl.after(customLauncherScript);

        window.zESettings = {
          webWidget: {
            chat: {
              departments: {
                enabled: [],
                select: 'Support',
              },
            },
          },
        };
      }
    })
    .then(() => {
      debugLog('Zendesk Chat initialized');
    });
}

// The default Intercom script for Weave
// For this intercom app https://app.intercom.com/a/apps/k3vi0wc0/messenger/web
const WEAVE_INTERCOM_SCRIPT = `(
  function(){
    var w=window;
    var ic=w.Intercom;
    if(typeof ic==="function"){
      ic('reattach_activator');
      ic('update',w.intercomSettings);
    }else{
      var d=document;
      var i=function(){
        i.c(arguments);
      };
      i.q=[];
      i.c=function(args){
        i.q.push(args);
      };
      w.Intercom=i;
      var l=function(){
        var s=d.createElement('script');
        s.type='text/javascript';
        s.async=true;
        s.src='https://widget.intercom.io/widget/k3vi0wc0';
        var x=d.getElementsByTagName('script')[0];
        x.parentNode.insertBefore(s,x);
      };
      if(document.readyState==='complete'){
        l();
      }else if(w.attachEvent){
        w.attachEvent('onload',l);
      }else{
        w.addEventListener('load',l,false);
      }
    }
  })();`;

// The default Intercom script for Automations
// For this intercom app https://app.intercom.com/a/apps/y230nsyv/messenger/web
const AUTOMATIONS_INTERCOM_SCRIPT = `(
  function(){
    var w=window;
    var ic=w.Intercom;
    if(typeof ic==="function"){
      ic('reattach_activator');
      ic('update',w.intercomSettings);
    }else{
      var d=document;
      var i=function(){
        i.c(arguments);
      };
      i.q=[];
      i.c=function(args){
        i.q.push(args);
      };
      w.Intercom=i;
      var l=function(){
        var s=d.createElement('script');
        s.type='text/javascript';
        s.async=true;
        s.src='https://widget.intercom.io/widget/y230nsyv';
        var x=d.getElementsByTagName('script')[0];
        x.parentNode.insertBefore(s,x);
      };
      if(document.readyState==='complete'){
        l();
      }else if(w.attachEvent){
        w.attachEvent('onload',l);
      }else{
        w.addEventListener('load',l,false);
      }
    }
  })();`;

/// Intercom Chat
function enableIntercomChat() {
  // Don't load on kaggle.com, colab.research.google.com, and iOS/Android
  // (behavior borrowed from enableZendeskChat which borrowed from Intercom)
  const kaggleRegex = /kaggle\.com/;
  const colabRegex = /colab\.research\.google\.com/;
  const mobileRegex = /iPhone|iPad|Android/;
  const isKaggle = kaggleRegex.test(window.location.host);
  const isColab = colabRegex.test(window.location.host);
  const isMobile = mobileRegex.test(navigator.userAgent);
  if (isKaggle || isColab || isMobile) {
    return;
  }

  for (const scriptContent of [WEAVE_INTERCOM_SCRIPT, AUTOMATIONS_INTERCOM_SCRIPT]) {
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.textContent = scriptContent;
    document.head.appendChild(script);
  }
}

function privateAnalyticsMain() {
  enableChameleon();
  enableDatadog();
}

function createAnalyticsGlobalInjector() {
  window.WBAnalyticsInjector = (function () {
    const scriptMap = {};

    function executeCookieControlledCallback(cookieControlGroupID, callback) {
      // Use Optanon to conditionally execute a callback. This is the "Onetrust"
      // cookie consent platform (https://app.onetrust.com) that we use to get
      // cookie consent. Cookies can be enabled/disabled by
      // category. Categories are configured in the OneTrust UI. cookieControlGroupID
      // specifies the category.

      // `callback` is only executed if the category is not disabled
      window.Optanon.InsertHtml(
        '<></>',
        'body',
        callback,
        null,
        cookieControlGroupID,
        false
      );
    }

    function executeConsentScripts(isAuthenticated = false) {
      // Iterate through scriptMap, executing the callback for each script
      // if the corresponding cookie consent category is enabled.
      const allProms = Object.entries(scriptMap).map(
        ([cookieControlGroupID, scriptDetails]) => {
          const {callback, hasRun} = scriptDetails;
          if (hasRun) {
            return Promise.resolve();
          }
          return new Promise((resolve, reject) => {
            function wrappedCallback() {
              // Somewhat obfuscated way of logging when a script is enabled
              // A: Enabled due to auth
              // OT: Enabled due to OneTrust
              debugLog(
                'CGID: ',
                cookieControlGroupID,
                ' enabled',
                isAuthenticated ? ' (Auth)' : ' (OneTrust)'
              );
              let res = callback();
              scriptDetails.hasRun = true;
              if (res instanceof Promise) {
                return res.then(resolve).catch(reject);
              }
              resolve();
            }
            if (isAuthenticated) {
              wrappedCallback();
            } else {
              executeCookieControlledCallback(
                cookieControlGroupID,
                wrappedCallback
              );
            }
          });
        }
      );
      return Promise.allSettled(allProms);
    }

    return {
      setScriptForGroupId: function (groupId, callback) {
        scriptMap[groupId] = {callback: callback, hasRun: false};
      },
      initializeTrackingScripts(isAuthenticated) {
        if (isAuthenticated) {
          return executeConsentScripts(isAuthenticated);
        }
        return new Promise((resolve, reject) => {
          // This is the callback that is executed when the user consents to
          // third-party analytics. It is called by OneTrust.
          window.OptanonWrapper = function OptanonWrapper() {
            executeConsentScripts().then(resolve).catch(reject);
          };
          // OneTrust Cookies Consent Notice
          // OneTrust requires JQuery, and by default loads an outdated version along with its own script.
          // We're turning off that default behavior and loading our own updated version of JQuery instead.
          // We must ensure that JQuery is fully loaded before inserting the OneTrust script
          // to avoid potential timing issues.
          insertScript(
            (window.CONFIG?.PUBLIC_URL ?? '') + '/jquery-3.6.0.min.js'
          ).then(() => {
            const suffix =
              window.CONFIG?.ENVIRONMENT_NAME === 'production' ? '' : '-test';
            const domainId = '29d3f242-6917-42f4-a828-bac6fba2e677' + suffix;

            insertScript(
              'https://cdn.cookielaw.org/consent/' + domainId + '/otSDKStub.js',
              {
                type: 'text/javascript',
                charset: 'UTF-8',
                'data-domain-script': domainId,
              }
            );
          });
        });
      },
    };
  })();
}

function setupAnalyticsMain() {
  window.thirdPartyAnalyticsOK = false;
  createAnalyticsGlobalInjector();

  // Concent-group specific scripts
  //   window.WBAnalyticsInjector.setScriptForGroupId('C0001', () => {});
  window.WBAnalyticsInjector.setScriptForGroupId('C0002', () => {
    const proms = [];
    window.thirdPartyAnalyticsOK = true;
    // Google analytics is synchronous
    enableGoogleTagManager();
    proms.push(enableSegmentAnalytics());
    proms.push(enableZendeskChat());
    proms.push(enableWebSights());
    proms.push(enableMixpanel());
    // Datadog is synchronous
    enableDatadog();
    enableIntercomChat();
    return Promise.allSettled(proms);
  });
  window.WBAnalyticsInjector.setScriptForGroupId('C0003', () => {
    const proms = [];
    proms.push(enableBeamerFeatureUpdates());
    return Promise.allSettled(proms);
  });


  //   window.WBAnalyticsInjector.setScriptForGroupId('C0004', () => {});
  //   window.WBAnalyticsInjector.setScriptForGroupId('C0005', () => {});
}

function main() {
  // Initialize Segment Methods
  addSegmentAnalyticsMethods();

  // Set analyticsOK to enable collection of local analytics
  window.analyticsOK = !window.CONFIG?.DISABLE_TELEMETRY;
  // Enable optanon cookie consent if third-party analytics are enabled. Optanon controls
  // inserting all of the other scripts.
  // Important we must set window.thirdPartyAnalyticsOK for the rest of the app
  window.thirdPartyAnalyticsOK = !window.CONFIG?.ANALYTICS_DISABLED;

  const shouldInitializeAnalytics =
    window.thirdPartyAnalyticsOK && !window.initializedScripts;
  if (shouldInitializeAnalytics) {
    setupAnalyticsMain();
  } else {
    const shouldInitializePrivateAnalytics =
      window.CONFIG?.ENVIRONMENT_IS_PRIVATE &&
      window.analyticsOK &&
      !window.initializedScripts;
    if (shouldInitializePrivateAnalytics) {
      privateAnalyticsMain();
    }
  }
  window.initializedScripts = true;
}

main();
