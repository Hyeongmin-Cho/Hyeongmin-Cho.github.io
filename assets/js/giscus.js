function loadGiscus() {
  const giscusContainer = document.querySelector('.giscus');
  if (giscusContainer) {
    giscusContainer.innerHTML = ''; // 기존 giscus DOM 제거
    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.setAttribute('data-repo', 'Hyeongmin-Cho/Hyeongmin-Cho.github.io'); // repo 설정
    script.setAttribute('data-repo-id', 'R_kgDOL-Z9UQ'); // repo ID 설정
    script.setAttribute('data-category', 'General'); // category 설정
    script.setAttribute('data-category-id', 'DIC_kwDOL-Z9Uc4CkykS'); // category ID 설정
    script.setAttribute('data-mapping', 'pathname'); // URL 매핑 방식
    script.setAttribute('data-strict', '0');
    script.setAttribute('data-reactions-enabled', '1'); // 리액션 활성화
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', 'bottom');
    script.setAttribute('data-lang', 'ko');
    script.setAttribute('data-theme', 'light'); // 테마 설정
    script.setAttribute('crossorigin', 'anonymous');
    script.async = true;
    giscusContainer.appendChild(script);
  }
}

// 페이지 로드 및 SPA 페이지 전환 감지
document.addEventListener('DOMContentLoaded', loadGiscus);
document.addEventListener('pjax:complete', loadGiscus);
