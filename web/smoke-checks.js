(async () => {
 const checks = [];
 const check = (name, value) => { if (!value) throw new Error(name); checks.push(name); };
 check('map loaded', ready && map.loaded());
 check('default counts preserved', document.getElementById('facts').textContent.includes('334,021'));
 check('advanced controls collapsed', !document.querySelector('.advanced').open);
 document.querySelector('.advanced').open = true;
 check('expanded controls fit viewport', document.documentElement.scrollWidth <= innerWidth);
 document.querySelector('.advanced').open = false;
 qi.value = 'Venice'; qi.dispatchEvent(new Event('input', {bubbles:true}));
 await new Promise(r=>setTimeout(r,400));
 check('search suggestions', !sg.hidden && sg.children.length > 0);
 qi.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
 check('place card opens', !document.getElementById('detail').hidden);
 document.getElementById('reset-explore').click();
 check('reset closes card', document.getElementById('detail').hidden);
 startYear.value = '4'; startYear.dispatchEvent(new Event('change'));
 check('date selection', lo === 4 && document.getElementById('range').textContent.includes('1550'));
 document.querySelector('#kinds button').click();
 check('place type filter', !kindOn.c);
 document.getElementById('reset-explore').click();
 check('reset dates and types', lo === 0 && hi === 9 && Object.values(kindOn).every(Boolean));
 document.querySelector('[data-r=aug]').click();
 await loadModel(); await new Promise(r=>setTimeout(r,400));
 check('expanded sources loaded', aug && MP.length > 0);
 document.querySelector('[data-r=strict]').click();
 check('editor sources restored', !aug && document.getElementById('facts').textContent.includes('334,021'));
 document.querySelector('[data-m=letters]').click();
 check('letters mode', mode === 'letters' && LVIS.length > 0);
 let opened = false;
 for (const [e] of LVIS) {                       // LVIS entries are [edge, weight]; open the card of an endpoint that has letters
   for (const p of [e.f, e.t]) {
     selectLetterPlace(p, false);
     const s = document.querySelector('#detail details.sec[data-sec=letters]');
     if (s) { s.open = true; s.dispatchEvent(new Event('toggle', {bubbles:true})); opened = true; break; }
   }
   if (opened) break;
 }
 await new Promise(r=>setTimeout(r,900));
 const ldocs = document.getElementById('docs');
 check('letters documents load', opened && !!ldocs && ldocs.children.length > 0 && !ldocs.textContent.includes('could not be loaded'));
 closeCard();
 document.querySelector('[data-m=places]').click();
 document.querySelector('[data-v=heat]').click();
 check('density mode', view === 'heat');
 document.querySelector('[data-v=circles]').click();
 document.querySelector('.header-link').click();
 check('about opens', document.getElementById('about-atlas').open);
 document.getElementById('about-atlas').open = false;
 document.getElementById('reset-explore').click(); window.scrollTo(0,0);
 check('no horizontal overflow', document.documentElement.scrollWidth <= innerWidth);
 return checks;
})()
