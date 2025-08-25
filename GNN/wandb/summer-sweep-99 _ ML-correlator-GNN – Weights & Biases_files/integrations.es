(function(FS) {
  var session = FS.getCurrentSession(), sessionUrl = FS.getCurrentSessionURL();
  function retryWithBackoff(condition, callback, maxWait, failureMsg, timeoutCallback) {
    var totalTimeExpired = 0;
    var wait = 64;
    var resultFn = function() {
      if (condition()) {
        callback();
        return;
      }
      wait = Math.min(wait * 2, 1024);
      if (totalTimeExpired > maxWait) {
        FS.log('warn', failureMsg);
        !!timeoutCallback && timeoutCallback(failureMsg);
        return;
      }
      totalTimeExpired += wait
      setTimeout(resultFn, wait);
    };
    return resultFn;
  }
  function loadSession(key) {
	var lastSession = window['localStorage'].getItem(key);
    if (!lastSession) {
      lastSession = FS._cookies()[key];
    }
    return lastSession
  }
  function saveSession(key, session) {
    window['localStorage'].setItem(key, session);
  }

	var mixpanelSessionUrl = sessionUrl + '?integration_src=mixpanel';
	// Check if the mixpanel script is on page
	if (window['mixpanel'] && typeof(mixpanel) == "object") {
		// Check if a new session has started since we last sent data to Mixpanel
		var lastSession = loadSession('fs_mixpanel');
		if (session != lastSession) {
			mixpanelFsIntegrationAsync();
		}
	} else {
		FS.log("warn", "Mixpanel integration is enabled, but mixpanel was not found on page");
		FS('stat', {
			eventType: 'INTEGRATION_INITIALIZED',
			payload: {
				provider_id: 'mixpanel',
				event_status: 'FAILURE',
				metadata: { detail: 'Mixpanel integration is enabled, but mixpanel was not found on page' }
			}
		})
	}

	// Wait for Mixpanel to be fully loaded before we try sending data
	// We can't use their on load callback because we don't call Mixpanel's init function, the client does that
	function mixpanelFsIntegrationAsync() {
		if(window['mixpanel']['__loaded']) {
			saveSession('fs_mixpanel', session);
			mixpanel.register({
				"FullStory Session": mixpanelSessionUrl
			});
			FS('stat', {
				eventType: 'INTEGRATION_INITIALIZED',
				payload: {
					provider_id: 'mixpanel',
					event_status: 'SUCCESS',
					metadata: { detail: 'mixpanel.register', sessionUrl: mixpanelSessionUrl }
				}
			})
			
		} else {
			setTimeout(mixpanelFsIntegrationAsync, 500);
		}
	}
	
})(window['_fs_namespace'] ? window[window['_fs_namespace']] : window['FS'])