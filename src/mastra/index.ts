import { Mastra } from "@mastra/core";
import { google } from "@ai-sdk/google";
import { Agent } from "@mastra/core/agent";
import { PgVector } from "@mastra/pg";
import { MDocument, createGraphRAGTool } from "@mastra/rag";

type Environment = {
	google_generative_ai_api_key: string;
	postgres_connection_string: string;
};

const parseEnv = (): Environment => {
	const e = {
		google_generative_ai_api_key: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
		postgres_connection_string: process.env.POSTGRES_CONNECTION_STRING,
	};
	if (!e.google_generative_ai_api_key || !e.postgres_connection_string) {
		throw new Error("Missing environment variables");
	}
	return e as Environment;
};

const env = parseEnv();
const embeddingModel = google.textEmbeddingModel("gemini-embedding-exp-03-07", {
	taskType: "RETRIEVAL_QUERY", // default value
});
const vectorStoreName = "pgVector";
const indexName = "embeddings";
const dimension = 768;
const graphRagTool = createGraphRAGTool({
	vectorStoreName,
	indexName,
	model: embeddingModel,
	graphOptions: {
		dimension,
		threshold: 0.7,
	},
});

const ragAgent = new Agent({
	name: "GraphRAG Agent",
	instructions: `You are a helpful assistant that answers questions based on the provided context. Format your answers as follows:

1. DIRECT FACTS: List only the directly stated facts from the text relevant to the question (2-3 bullet points)
2. CONNECTIONS MADE: List the relationships you found between different parts of the text (2-3 bullet points)
3. CONCLUSION: One sentence summary that ties everything together

Keep each section brief and focus on the most important points.

Important: When asked to answer a question, please base your answer only on the context provided in the tool.
If the context doesn't contain enough information to fully answer the question, please state that explicitly.`,
	model: google.chat("gemini-2.0-flash"),
	tools: {
		graphRagTool,
	},
});

const pgVector = new PgVector({
	connectionString: env.postgres_connection_string,
});

export const mastra = new Mastra({
	agents: { ragAgent },
	vectors: { pgVector },
});

const html = `<!DOCTYPE HTML>
<html lang="ja">
<head>
  <meta charset="utf-8">


        <meta name="keywords" content="">
<meta name="description" content="">    <meta property="og:title" content="家庭ごみの分け方と出し方及びごみ収集計画表|氷見市">
<meta property="og:type" content="article">
<meta property="og:url" content="https://www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/1/3098.html">
  <meta property="og:image" content="http://www.city.himi.toyama.jp/material/images/group/18/gomi_bunbetsu_medium.png" />
  <meta property="og:description" content="               家庭ごみの分け方と出し方は、下記の添付ファイルをご参照ください。            " />  <meta name="viewport" content="width=750, user-scalable=yes">      <meta name="nsls:timestamp" content="Wed, 24 Jan 2024 23:30:00 GMT">            <title>家庭ごみの分け方と出し方及びごみ収集計画表／氷見市</title>                          <link rel="canonical" href="https://www.city.himi.toyama.jp/gyosei/soshiki/kankyo/1/1/1931.html">
                            <link rel="icon" href="//www.city.himi.toyama.jp/favicon.ico">
        <link rel="apple-touch-icon" href="//www.city.himi.toyama.jp/theme/base/img_common/smartphone.png">
                                                  <link href="//www.city.himi.toyama.jp/theme/base/css/sub.css" rel="stylesheet" type="text/css" class="sp-style">



                                                        <script src="//www.city.himi.toyama.jp/theme/base/js/jquery.js"></script>
                              <script src="//www.city.himi.toyama.jp/theme/base/js/jquery_cookie.js"></script>
                              <script src="//www.city.himi.toyama.jp/theme/base/js/jquery-ui.min.js"></script>
                              <script src="//www.city.himi.toyama.jp/theme/base/js/common_lib.js"></script>
                              <script src="//www.city.himi.toyama.jp/theme/base/js/jquery.easing.1.3.js"></script>
                              <script src="//www.city.himi.toyama.jp/theme/base/js/jquery.bxslider.js"></script>
                              <script src="//www.city.himi.toyama.jp/theme/base/js/jquery_dropmenu.js"></script>

<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-NKSRR8V');</script>

<script src="//www.city.himi.toyama.jp/theme/base/js/common.js"></script>
<link href="//www.city.himi.toyama.jp/theme/base/css/vegas.min.css" rel="stylesheet" type="text/css">
<script src="//www.city.himi.toyama.jp/theme/base/js/vegas.min.js"></script>
<script src="//www.city.himi.toyama.jp/theme/base/js/mutual_switching/mutual_switching.js"></script>
                                              <script src='//www.google.com/jsapi'></script>
                                          <script src="//www.city.himi.toyama.jp/theme/base/js/sub.js"></script>




  <!--[if lt IE 9]>
  <script src="//www.city.himi.toyama.jp/theme/base/js/html5shiv-printshiv.min.js"></script>
  <script src="//www.city.himi.toyama.jp/theme/base/js/css3-mediaqueries.js"></script>
  <![endif]-->

  <script>
    var cms_api_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJjdXN0b21lcl9jb2RlIjoiMTkxMTgwIiwic2VydmljZV9uYW1lIjoiU01BUlQgQ01TIn0.vdYjv_KeypyZ0voHuT5NOcn_yQ8fu8edVLq1dv3uSyQ";
    var cms_api_domain="api4th.smart-lgov.jp";
    var cms_api_site="";
    var cms_app_version="";
    var cms_app_id="";
    var site_domain = "https://www.city.himi.toyama.jp";
    var theme_name = "base";
    var cms_recruit_no = "0";
    var cms_recruit_history_no = "0";
    var cms_recruit_search_item = '[]';
    var is_smartphone = false;  </script>





</head>
<body>





  <div id="wrapper">
    <div id="wrapper-in">
      <div id="wrapper-in2">

        <div id="header-print">
                      <script>
$(function() {
  $('.headerNaviDynBlock').each(function() {
    var block = $(this);
    var list = block.find('.headerNaviDynList');
    block.css('display', 'none');

    var url = block.attr('url');
    if (!url) {
      url = block.attr('data-url');
      if (!url) {
        return;
      }
    }

    $.getJSON(url, function(json) {
      var templateOrig = block.find('.headerNaviPageTemplate');
      if (templateOrig.length == 0) {
        return;
      }
      var template = templateOrig.clone().removeClass('headerNaviPageTemplate').addClass('pageEntity').css('display', '');
      block.find('.pageEntity').remove();
      var count = 0;
      for (var j=0; j<json.length; j++) {
        var item = json[j];
        if (item.is_category_index && item.child_pages_count == 0) {
          continue;
        }
        var entity = template.clone();
        entity.find('.pageLink').attr('href', item.url).text(item.page_name);
        entity.find('.pageDescription').text(item.description);
        list.append(entity);
        count++;
      }
      if (count > 0) {
        block.css('display', '');
      }
      templateOrig.remove();
    });
  });
});
</script>
<p id="smartphone" class="jqs-go-to-sp" style="display: none;">
  <a href="https://www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/1/3098.html" class="jqs-go-to-sp">
    <img src="//www.city.himi.toyama.jp/theme/base/img_common/btn_display_smartphone.png" alt="スマートフォン版を表示">
  </a>
</p>

<p class="to-container"><a href="#container"><span>本文へ</span></a></p>
<header id="header" class="view-pc">
  <div class="header-in">
    <div class="header-subnav-area">
      <nav class="unit">
        <ul class="list">
          <li><a href="https://www.city.himi.toyama.jp/gyosei/soshiki/index.html">組織から探す</a></li>
          <li><a href="https://www.city.himi.toyama.jp/sitemap/index.html">サイトマップ</a></li>
        </ul>
      </nav>
      <dl id="header-size">
        <dt class="title"><span>文字サイズ</span></dt>
        <dd class="item"><a href="#" class="scsize normal"><img src="//www.city.himi.toyama.jp/theme/base/img_common/headersize_normal_on.png" alt="標準"></a></dd>
        <dd class="item2"><a href="#" class="scsize up"><img src="//www.city.himi.toyama.jp/theme/base/img_common/headersize_big_off.png" alt="拡大"></a></dd>
      </dl>
      <dl id="header-color">
        <dt class="title"><span>背景色変更</span></dt>
        <dd class="item"><a href="#" class="sccolor" data-bgcolor="color_black"><img src="//www.city.himi.toyama.jp/theme/base/img_common/headercolor_black.png" alt="背景色を黒色にする"></a></dd>
        <dd class="item2"><a href="#" class="sccolor" data-bgcolor="color_blue"><img src="//www.city.himi.toyama.jp/theme/base/img_common/headercolor_blue.png" alt="背景色を青色にする"></a></dd>
        <dd class="item3"><a href="#" class="sccolor" data-bgcolor="color_normal"><img src="//www.city.himi.toyama.jp/theme/base/img_common/headercolor_white.png" alt="背景色を白色にする"></a></dd>
      </dl>
      <div class="unit2">
        <ul class="list">
          <li lang="en"><a href="https://www.city.himi.toyama.jp/gyosei/5429.html">Select Language</a></li>
        </ul>
      </div>
    </div>
          <p id="header-logo">
        <a href="https://www.city.himi.toyama.jp/gyosei/index.html"><img src="//www.city.himi.toyama.jp/theme/base/img_common/pc_header_logo.png" alt="氷見市 HIMI CITY"></a>
      </p>
        <nav id="header-nav">
      <ul class="list">
        <li class="nav1">
          <a href="https://www.city.himi.toyama.jp/gyosei/kurashi/index.html">くらし・生活</a>
                    <div class="dropmenu menu-1">
            <div class="dropmenu-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/kurashi/index.tree.json">
              <ul class="list2 headerNaviDynList">
                <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
              </ul>
            </div>
          </div>
                  </li>
        <li class="nav2">
          <a href="https://www.city.himi.toyama.jp/gyosei/bosai_syobo/index.html">防災・安全・消防</a>
                    <div class="dropmenu menu-2">
            <div class="dropmenu-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/bosai_syobo/index.tree.json">
              <ul class="list2 headerNaviDynList">
                <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
              </ul>
            </div>
          </div>
                  </li>
        <li class="nav3">
          <a href="https://www.city.himi.toyama.jp/gyosei/bunka/index.html">市民活動・<br>文化活動</a>
                    <div class="dropmenu menu-3">
            <div class="dropmenu-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/bunka/index.tree.json">
              <ul class="list2 headerNaviDynList">
                <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
              </ul>
            </div>
          </div>
                  </li>
        <li class="nav4">
          <a href="https://www.city.himi.toyama.jp/kosodate/index.html">子育て・教育</a>
        </li>
        <li class="nav5">
          <a href="https://www.city.himi.toyama.jp/gyosei/iryo/index.html">医療・福祉</a>
                    <div class="dropmenu menu-5">
            <div class="dropmenu-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/iryo/index.tree.json">
              <ul class="list2 headerNaviDynList">
                <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
              </ul>
            </div>
          </div>
                  </li>
        <li class="nav6">
          <a href="https://www.city.himi.toyama.jp/gyosei/bijinesu_sangyo/index.html">ビジネス・産業</a>
                    <div class="dropmenu menu-6">
            <div class="dropmenu-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/bijinesu_sangyo/index.tree.json">
              <ul class="list2 headerNaviDynList">
                <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
              </ul>
            </div>
          </div>
                  </li>
        <li class="nav7">
          <a href="https://www.city.himi.toyama.jp/gyosei/shisei/index.html">市政情報</a>
                    <div class="dropmenu menu-7">
            <div class="dropmenu-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/shisei/index.tree.json">
              <ul class="list2 headerNaviDynList">
                <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
              </ul>
            </div>
          </div>
                  </li>
      </ul>
    </nav>
  </div>
</header>
<header id="sp-header" class="view-sp">
  <div class="box clearfix">
    <p id="sp-header-logo">
      <a href="https://www.city.himi.toyama.jp/gyosei/index.html">
        <img src="//www.city.himi.toyama.jp/theme/base/s-admin/img_common/sp_header_logo.png" alt="氷見市 HIMI CITY">
      </a>
    </p>
    <nav id="sp-header-nav" class="clearfix">
      <div class="menu-btn-area clearfix">
      <p class="menu-btn-menu"><a href="#"><img src="//www.city.himi.toyama.jp/theme/base/s-admin/img_common/menu_btn_menu.png" alt="メニュー"></a></p>
      </div>
    </nav>
  </div>
  <div class="modal-menu">
    <div class="in">
      <div class="modal-menu-close top-btn-close">
        <a href="#"><img src="//www.city.himi.toyama.jp/theme/base/s-admin/img_common/menu_btn_close.png" alt="メニューを閉じる"></a>
      </div>
      <div class="keyword-area">
        <h3>キーワード検索</h3>









                                      <div class="gcse-searchbox-only" data-resultsurl="//www.city.himi.toyama.jp/result.html" data-enableAutoComplete="true"></div>

                      </div>
      <div class="box">
                <h2 class="title"><a href="#">くらし・生活</a></h2>
        <div class="list-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/kurashi/index.tree.json">
          <ul class="list headerNaviDynList">
            <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
          </ul>
        </div>
              </div>
      <div class="box">
                <h2 class="title"><a href="#">防災・安全・消防</a></h2>
        <div class="list-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/bosai_syobo/index.tree.json">
          <ul class="list headerNaviDynList">
            <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
          </ul>
        </div>
              </div>
      <div class="box">
                <h2 class="title"><a href="#">市民活動・文化活動</a></h2>
        <div class="list-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/bunka/index.tree.json">
          <ul class="list headerNaviDynList">
            <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
          </ul>
        </div>
              </div>
      <div class="box">
        <div class="title-link"><a href="https://www.city.himi.toyama.jp/kosodate/index.html">子育て・教育</a></div>
      </div>
      <div class="box">
                <h2 class="title"><a href="#">ビジネス・産業</a></h2>
        <div class="list-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/bijinesu_sangyo/index.tree.json">
          <ul class="list headerNaviDynList">
            <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
          </ul>
        </div>
              </div>
      <div class="box">
                <h2 class="title"><a href="#">医療・福祉</a></h2>
        <div class="list-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/iryo/index.tree.json">
          <ul class="list headerNaviDynList">
            <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
          </ul>
        </div>
              </div>
      <div class="box">
                <h2 class="title"><a href="#">市政情報</a></h2>
        <div class="list-in headerNaviDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/shisei/index.tree.json">
          <ul class="list headerNaviDynList">
            <li class="headerNaviPageTemplate"><a class="pageLink"></a></li>
          </ul>
        </div>
              </div>
      <div class="link-pc">
        <p class="menu-btn-pc jqs-go-to-pc">
          <a href="https://www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/1/3098.html" class="jqs-go-to-pc">PCサイト版を表示</a>
        </p>
      </div>
      <div class="box-2 clearfix">
        <dl class="header-size">
          <dt class="size-title"><span>文字サイズ</span></dt>
          <dd class="item"><a href="#" class="scsize normal"><img src="//www.city.himi.toyama.jp/theme/base/img_common/sp_headersize_normal_on.png" alt="標準"></a></dd>
          <dd class="item2"><a href="#" class="scsize up"><img src="//www.city.himi.toyama.jp/theme/base/img_common/sp_headersize_big_off.png" alt="拡大"></a></dd>
        </dl>
        <dl class="header-color">
          <dt class="color-title"><span>背景色変更</span></dt>
          <dd class="item"><a href="#" class="sccolor" data-bgcolor="color_black"><img src="//www.city.himi.toyama.jp/theme/base/img_common/sp_headercolor_black.png" alt="背景色を黒色にする"></a></dd>
          <dd class="item2"><a href="#" class="sccolor" data-bgcolor="color_blue"><img src="//www.city.himi.toyama.jp/theme/base/img_common/sp_headercolor_blue.png" alt="背景色を青色にする"></a></dd>
          <dd class="item3"><a href="#" class="sccolor" data-bgcolor="color_normal"><img src="//www.city.himi.toyama.jp/theme/base/img_common/sp_headercolor_white.png" alt="背景色を白色にする"></a></dd>
        </dl>
        <div class="language" lang="en">
          <div class="language-btn">
            <a href="https://www.city.himi.toyama.jp/gyosei/5429.html">Select Language</a>
          </div>
        </div>
      </div>
      <div class="modal-menu-close bottom-btn-close">
        <p><a href="#"><img src="//www.city.himi.toyama.jp/theme/base/img_top/btn-menu-close.png" alt="メニューを閉じる"></a></p>
      </div>
    </div>
  </div>
</header>                  </div>

        <section id="container">
          <div id="container-in" class="clearfix">

            <dl id="pankuzu" class="clearfix">
              <dt class="title">現在の位置</dt>
              <dd class="in">


<ul class="list">
              <li><a href="https://www.city.himi.toyama.jp/index.html">総合トップ</a></li>
                  <li class="icon"><a href="https://www.city.himi.toyama.jp/gyosei/index.html">氷見市ホームページ</a></li>
                  <li class="icon"><a href="https://www.city.himi.toyama.jp/gyosei/kurashi/index.html">くらし・生活</a></li>
                  <li class="icon"><a href="https://www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/index.html">ごみ・環境保全</a></li>
                  <li class="icon"><a href="https://www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/1/index.html">ごみの出し方</a></li>
            <li class="icon"><span>家庭ごみの分け方と出し方及びごみ収集計画表</span></li>
  </ul>
              </dd>
            </dl>

            <article id="contents" role="main">


      <h1 class="title"><span class="bg"><span class="bg2">家庭ごみの分け方と出し方及びごみ収集計画表</span></span></h1>

              <div id="social-update-area">







        <p class="update">更新日：2024年01月25日</p>


              </div>



              <div id="contents-in">



                <div class="free-layout-area">
          <div>











            <h2><span class="bg"><span class="bg2"><span class="bg3">家庭ごみの分け方と出し方</span></span></span></h2>
























            <div class="wysiwyg">
              <p>&nbsp;家庭ごみの分け方と出し方は、下記の<strong>添付ファイル</strong>をご参照ください。</p>
            </div>












<p class="file-link-item"><a class="pdf" href="//www.city.himi.toyama.jp/material/files/group/18/65294864.pdf">1803氷見市(WEB用) (PDFファイル: 6.6MB)</a></p>












            <h2><span class="bg"><span class="bg2"><span class="bg3">収集計画</span></span></span></h2>
























            <div class="wysiwyg">
              <p>令和7年度及び令和6年度の燃やせるごみ、燃やせないごみ及び資源ごみの収集計画は、下記の<strong>添付ファイル</strong>をご参照ください。</p>

<p><a target="_blank" class="icon2" href="//www.city.himi.toyama.jp/material/files/group/18/R7gomisyuusyuukeikaku.pdf">令和7年度ごみ収集計画表(PDFファイル:150.8KB)</a></p>

<p><a target="_blank" class="icon2" href="//www.city.himi.toyama.jp/material/files/group/18/R6gomisyuusyuukeikaku.pdf">令和6年度ごみ収集計画表(PDFファイル:168.4KB)</a></p>
            </div>
























            <h3><span class="bg"><span class="bg2"><span class="bg3">家庭系燃やせるごみの指定袋について</span></span></span></h3>
























            <div class="wysiwyg">
              <p>高岡地区広域圏事務組合の共通指定袋（乳白色）をお使いください。</p>

<ul>
	<li>袋の大きさや販売価格<br />
	大：45リットル10枚入り300円、中：20リットル10枚入り200円、小：10リットル10枚入り100円</li>
</ul>

<p>(注意)すでにお買い求めの赤色の氷見市指定袋は、そのままお使いいただくことができます。</p>
            </div>
























            <h3><span class="bg"><span class="bg2"><span class="bg3">家庭系燃やせないごみの袋について</span></span></span></h3>
























            <div class="wysiwyg">
              <p>無色透明の袋をお使いください。</p>
            </div>
























            <h3><span class="bg"><span class="bg2"><span class="bg3">分別収集する（プラスチック製容器包装）ごみ</span></span></span></h3>
























            <div class="wysiwyg">
              <p>緑色の袋をお使いください。</p>
            </div>
























            <h2><span class="bg"><span class="bg2"><span class="bg3">持込みごみの搬入方法について</span></span></span></h2>
























            <div class="wysiwyg">
              <p>燃やせるごみを持ち込む場合は、「<a class="icon" href="https://www.city.himi.toyama.jp/gyosei/soshiki/kankyo/1/1/1932.html">燃やせる持込みごみの搬入方法について</a>」をご確認のうえ事前にお問い合わせください。</p>

<p>燃やせないごみを持ち込む場合は、「<a class="icon" href="https://www.city.himi.toyama.jp/gyosei/soshiki/kankyo/1/1/1934.html">燃やせない持込みごみの搬入方法について</a>」をご確認のうえ事前にお問い合わせください。</p>

<p>資源化するごみ（リサイクル販売品）を持ち込む場合は、「<a class="icon" href="https://www.city.himi.toyama.jp/gyosei/soshiki/kankyo/1/5/1935.html">資源化する持込みごみ（リサイクル販売品）の搬入方法について</a>」をご確認のうえ事前にお問い合わせください。</p>
            </div>
























            <figure class="img-item"><img alt="女性が資源ごみ、もえるごみ、もえないごみをごみ箱へ入れているイラスト" src="//www.city.himi.toyama.jp/material/images/group/18/gomi_bunbetsu_medium.png"></figure>












</div>
        </div>






                                                                  <!-- 「お問い合わせ先」 -->

      <div class="toiawase">
      <div class="toiawase-in">
        <h2 class="title"><span class="bg"><span class="bg2">この記事に関するお問い合わせ先</span></span></h2>
        <div class="in">
          <div class="name"><p>環境保全課<br><br>郵便番号：935-8686<br>富山県氷見市鞍川1060番地<br>電話番号：0766-74-8065 ファックス番号：0766-74-8104<br><a href="https://www.city.himi.toyama.jp/cgi-bin/inquiry.php/46?page_no=3098">メールでのお問い合わせはこちら</a></p></div>
        </div>
      </div>
    </div>

                     <!-- pdfダウンロード -->

          <aside id="pdf-download">
    <a href="http://get.adobe.com/jp/reader/">
    <img src="//www.city.himi.toyama.jp/resource/img/get_adobe_reader.png" alt="「Adobe Reader（Acrobat Reader）」ダウンロード"></a>
    PDFファイルを閲覧するには「Adobe Reader（Acrobat Reader）」が必要です。お持ちでない方は、左記の「Adobe Reader（Acrobat Reader）」ダウンロードボタンをクリックして、ソフトウェアをダウンロードし、インストールしてください。
  </aside>







              <!-- //#contents-in  -->
              </div>
            <!-- //#contents  -->
            </article>



                <nav id="side-nav">
  <div class="google">









                                      <div class="gcse-searchbox-only" data-resultsurl="//www.city.himi.toyama.jp/result.html" data-enableAutoComplete="true"></div>

              </div>

        <section class="side-nav-list">

    <script>
  function cmsDynDateFormat(date, format) {
    var jpWeek = ['日', '月', '火', '水', '木', '金', '土'];
    return format.replace('%Y', date.getFullYear()).replace('%m', ('0' + (date.getMonth() + 1)).slice(-2)).replace('%d', ('0' + date.getDate()).slice(-2)).replace('%a', jpWeek[date.getDay()])
        .replace('%H', ('0' + date.getHours()).slice(-2)).replace('%M', ('0' + date.getMinutes()).slice(-2)).replace('%S', ('0' + date.getSeconds()).slice(-2));
  }
  function cmsDynExecuteGetPageList() {
    var outerBlocks = $('.pageListDynBlock');
    outerBlocks.each(function() {
      var block = $(this);
      block.find('.pageListExists').css('display', 'none');
      block.find('.pageListNotExists').css('display', 'none');

      var url = block.attr('data-url');

      var cond = {};

      cond.limit = parseInt(block.attr('data-limit'));
      cond.showIndex = parseInt(block.attr('data-show-index'));
      cond.showMobile = parseInt(block.attr('data-show-mobile'));
      dateBegin = block.attr('data-date-begin');
      dateSpan = block.attr('data-date-span');

      cond.curPageNo = block.attr('data-current-page-no');
      cond.dirClass = block.attr('data-dir-class');
      cond.pageClass = block.attr('data-page-class');

      cond.timeBegin = 0;
      if (dateBegin) {
        cond.timeBegin = new Date(dateBegin);
      } else if (dateSpan) {
        cond.timeBegin = Date.now() - dateSpan * 86400000;
      }
      var recentSpan = block.attr('data-recent-span');
      cond.recentBegin = 0;
      if (recentSpan) {
        cond.recentBegin = Date.now() - recentSpan * 86400000;
      }
      cond.dateFormat = block.attr('data-date-format');
      if (!cond.dateFormat) {
        cond.dateFormat = '%Y/%m/%d %H:%M:%S';
      }
      cond.joinGrue = block.attr('data-join-grue');
      if (!cond.joinGrue) {
        cond.joinGrue = ' , ';
      }
      cond.eventDateFormat = block.attr('data-event-date-format');
      if (!cond.eventDateFormat) {
        cond.eventDateFormat = cond.dateFormat;
      }
      cond.eventType = block.attr('data-event-type');
      cond.eventField = block.attr('data-event-field');
      cond.eventArea = block.attr('data-event-area');
      eventDateSpan = block.attr('data-event-date-span');
      cond.eventTimeEnd = 0;
      if (eventDateSpan) {
        cond.eventTimeEnd = Date.now() + eventDateSpan * 86400000;
      }

      // タグ
      cond.tagDisplay = block.attr('data-show-tags');
      cond.tagPosition = block.attr('data-tags-position');
      cond.tagFilterTargets = block.attr('data-tag-filter-targets');

      $.getJSON(url, function(json) {
        cmsDynApplyPageListJson(block, json, cond);
      }).fail(function(jqxhr, textStatus, error) {
        block.css('display', 'none');
      });
    });
  }
  function cmsDynApplyPageListJson(block, json, cond) {
    var now = Date.now();
    var list = block.find('.pageListBlock');
    var template = list.find('.pageEntity:first').clone();
    list.find('.pageEntity').remove();

    var count = 0;

    for (var i = 0; i < json.length; i++) {
      var item = json[i];
      var itemDate = new Date(item.publish_datetime);

      if (!cond.showIndex && item.is_category_index) {
        continue;
      }
      if (!cond.showMobile && item.is_keitai_page) {
        continue;
      }
      if (cond.timeBegin && itemDate.getTime() < cond.timeBegin) {
        continue;
      }

      // タグによる絞込み
      if ('tag' in item && item.tag && cond.tagFilterTargets != null) {
        var filteringNos = (!isNaN(cond.tagFilterTargets)) ? [cond.tagFilterTargets] : cond.tagFilterTargets.split(/,|\s/);
        var isTarget = false;
        item.tag.forEach(function(tagItem, idx) {
          if (filteringNos.indexOf(tagItem.tag_no + "") >= 0) {
            isTarget = true;
          }
        });
        if (!isTarget) {
          continue;
        }
      }

      var entity = template.clone();
      if ('event' in item && item['event']) {
        var pageEvent = item['event'];
        if (cond.eventType && cond.eventType != pageEvent.event_type_name) {
          continue;
        }
        if (cond.eventField && $.inArray(cond.eventField, pageEvent.event_fields) < 0) {
          continue;
        }
        if (cond.eventArea && $.inArray(cond.eventArea, pageEvent.event_area) < 0) {
          continue;
        }

        var eventDateString = '';
        if (cond.eventTimeEnd) {
          if (pageEvent.event_date_type_id == 0) {
            var startDatetime = pageEvent.event_start_datetime ? new Date(pageEvent.event_start_datetime) : false;
            var endDatetime = pageEvent.event_end_datetime ? new Date(pageEvent.event_end_datetime) : false;
            if (startDatetime && endDatetime) {
              if (startDatetime.getTime() > cond.eventTimeEnd || endDatetime.getTime() <= now) {
                continue;
              }
              eventDateString = cmsDynDateFormat(startDatetime, cond.eventDateFormat) + '～' + cmsDynDateFormat(endDatetime, cond.eventDateFormat);
            } else if (startDatetime) {
              if (startDatetime.getTime() > cond.eventTimeEnd) {
                continue;
              }
            } else {
              if (endDatetime.getTime() <= now) {
                continue;
              }
              eventDateString = '～' + cmsDynDateFormat(endDatetime, cond.eventDateFormat);
            }
          } else if (pageEvent.event_date_type_id == 1) {
            var filteredDates = $.grep(pageEvent.event_dates, function(value, index) {
              var eventTime1 = new Date(value[0]+'T00:00:00+09:00').getTime();
              var eventTime2 = new Date(value[1]+'T23:59:59+09:00').getTime();
              return (eventTime1 <= cond.eventTimeEnd && eventTime2 >= now);
            });
            if (filteredDates.length == 0) {
              continue;
            }
          }
        }
        if (pageEvent.event_place) {
          entity.find('.pageEventPlaceExists').css('display', '');
          entity.find('.pageEventPlace').text(pageEvent.event_place);
        } else {
          entity.find('.pageEventPlaceExists').css('display', 'none');
          entity.find('.pageEventPlace').text('');
        }
        if (pageEvent.event_date_supplement) {
          entity.find('.pageEventDateExists').css('display', '');
          entity.find('.pageEventDate').text(pageEvent.event_date_supplement);
        } else if (eventDateString.length > 0) {
          entity.find('.pageEventDateExists').css('display', '');
          entity.find('.pageEventDate').text(eventDateString);
        } else {
          entity.find('.pageEventDateExists').css('display', 'none');
          entity.find('.pageEventDate').text('');
        }

        if (pageEvent.event_type_name) {
          entity.find('.pageEventTypeExists').css('display', '');
          entity.find('.pageEventType').text(pageEvent.event_type_name);
        } else {
          entity.find('.pageEventTypeExists').css('display', 'none');
          entity.find('.pageEventType').text('');
        }
        if (pageEvent.event_fields && pageEvent.event_fields.length > 0) {
          entity.find('.pageEventFieldsExists').css('display', '');
          entity.find('.pageEventFields').text(pageEvent.event_fields.join(cond.joinGrue));
        } else {
          entity.find('.pageEventFieldsExists').css('display', 'none');
          entity.find('.pageEventFields').text('');
        }
        if (pageEvent.event_area && pageEvent.event_area.length > 0) {
          entity.find('.pageEventAreaExists').css('display', '');
          entity.find('.pageEventArea').text(pageEvent.event_area.join(cond.joinGrue));
        } else {
          entity.find('.pageEventAreaExists').css('display', 'none');
          entity.find('.pageEventArea').text('');
        }
        entity.find('.pageEventExists').css('display', '');
      } else {
        entity.find('.pageEventExists').css('display', 'none');
      }

      entity.find('.pageDate').each(function() {
        var dateString = cmsDynDateFormat(itemDate, cond.dateFormat);
        $(this).text(dateString);
      });
      var pageLink = entity.find('a.pageLink');
      if (cond.curPageNo == item.page_no) {
        pageLink.removeAttr('href').removeAttr('page_no').css('display', 'none');
        pageLink.parent().append('<span class="pageNoLink">' + item.page_name + '</span>');
      } else {
        pageLink.attr('page_no', item.page_no).attr('href', item.url).text(item.page_name);
        pageLink.find('.pageNoLink').remove();
      }

      entity.find('.pageDescription').text(item.description);

      if ('thumbnail_image' in item && item.thumbnail_image) {
        entity.find('.pageThumbnail').append($('<img>', {src: item.thumbnail_image, alt: ""}));
      } else {
        entity.find('.pageThumbnail').remove();
      }

      if (cond.recentBegin && itemDate.getTime() >= cond.recentBegin) {
        entity.find('.pageRecent').css('display', '');
      } else {
        entity.find('.pageRecent').css('display', 'none');
      }

      // タグ付与
      if ('tag' in item && item.tag) {
        if (item.tag.length > 0) {
          var DEFINE_CLASS_NAME_WHEN_TAG_TYPE_IMAGE = 'tag-type-image';
          var DEFINE_CLASS_NAME_WHEN_TAG_TYPE_TEXT = 'tag-type-text';
          var DEFINE_CLASS_NAME_WHEN_TAG_POSITION_BEFORE = 'tag-pos-before';
          var DEFINE_CLASS_NAME_WHEN_TAG_POSITION_AFTER = 'tag-pos-after';
          var DEFINE_CLASS_NAME_TAG_BLOCK = 'tags';
          var DEFINE_CLASS_NAME_TAG = 'tag';
          var DEFINE_CLASS_NAME_TAG_INNER = 'tag-bg';

          // タグの表示位置を判定
          var tagPositionClassName = (cond.tagPosition == 1) ? DEFINE_CLASS_NAME_WHEN_TAG_POSITION_BEFORE : DEFINE_CLASS_NAME_WHEN_TAG_POSITION_AFTER;

          // タグ出力の外枠を生成
          var tagListWrapperHtml = $('<span>', {
            class: [DEFINE_CLASS_NAME_TAG_BLOCK, tagPositionClassName].join(' ')
          });

          item.tag.forEach(function(tagItem, idx) {
            // タグの中身を設定
            var tagBody;
            if (tagItem.image_file_name != null && tagItem.image_file_name != "") {
              // 画像
              tagBody = $('<span>', {
                class: DEFINE_CLASS_NAME_TAG + tagItem.tag_no,
              }).append($('<img>', {
                class: [DEFINE_CLASS_NAME_TAG_INNER, DEFINE_CLASS_NAME_WHEN_TAG_TYPE_IMAGE].join(' '),
                src: tagItem.image_url,
                alt: tagItem.tag_name
              }));
            } else {
              // テキスト
              tagBody = $('<span>', {
                class: DEFINE_CLASS_NAME_TAG + tagItem.tag_no,
              }).append($('<span>', {
                class: [DEFINE_CLASS_NAME_TAG_INNER, DEFINE_CLASS_NAME_WHEN_TAG_TYPE_TEXT].join(' '),
                text: tagItem.tag_name
              }));
            }
            tagListWrapperHtml.append(tagBody);
          });

          // 出力
          if (cond.tagDisplay == 1) {
            if (tagPositionClassName === DEFINE_CLASS_NAME_WHEN_TAG_POSITION_BEFORE) {
              entity.find('a.pageLink').before(tagListWrapperHtml);
            } else {
              entity.find('a.pageLink').after(tagListWrapperHtml);
            }
          }
        }
      }

      var removeClasses = [];
      var appendClasses = [];
      if (item.is_category_index) {
        appendClasses = cond.dirClass ? cond.dirClass.split(' ') : [];
        removeClasses = cond.pageClass ? cond.pageClass.split(' ') : [];
      } else {
        removeClasses = cond.dirClass ? cond.dirClass.split(' ') : [];
        appendClasses = cond.pageClass ? cond.pageClass.split(' ') : [];
      }
      $.each(removeClasses, function(idx, val){
        entity.removeClass(val);
      });
      $.each(appendClasses, function(idx, val){
        entity.addClass(val);
      });

      entity.css('display', '');
      list.append(entity);
      count++;
      if (cond.limit && count >= cond.limit) {
        break;
      }
    }
    if (count) {
      block.css('display', '');
      block.find('.pageListExists').css('display', '');
      block.find('.pageListNotExists').css('display', 'none');
    } else {
      block.css('display', '');
      block.find('.pageListExists').css('display', 'none');
      block.find('.pageListNotExists').css('display', '');
    }
  };
</script>

<script>
$(function() {
  cmsDynExecuteGetPageList();
});
</script>



  <div class="pageListDynBlock" data-url="//www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/1/index.tree.json"
   data-show-shortcut="1" data-show-index="1"
   data-current-page-no="3098">
    <dl class="pageListExists">
      <dt class="title">
        <span class="bg"><span class="bg2"><a href="//www.city.himi.toyama.jp/gyosei/kurashi/gomi_kankyo/1/index.html">ごみの出し方</a></span></span>
      </dt>
      <dd class="in">
        <ul class="list clearfix pageListBlock">
          <li class="pageEntity" style="display:none;">
            <a class="pageLink"></a>
          </li>
        </ul>
      </dd>
    </dl>
  </div>
    </section>


  </nav>
          <!-- //#container-in  -->
          </div>
        <!-- //#container  -->
        </section>

        <div id="footer-print">
                      <footer id="footer">
  <div class="in">
    <p class="pagetop"><a href="#wrapper" class="scroll">ページトップへ</a></p>
    <div class="box">
      <div class="logo" id="footer-logo">
        <span class="visually-hidden">氷見市 HIMI CITY</span>
        <img src="//www.city.himi.toyama.jp/theme/base/img_common/pc_footer_logo.png" alt="" class="view-pc">
        <img src="//www.city.himi.toyama.jp/theme/base/img_common/sp_footer_logo.png" alt="" class="view-sp">
      </div>
      <p class="name">氷見市役所</p>
      <div class="address">
        <p>〒935-8686　富山県氷見市鞍川1060番地</p>
        <p>電話：0766-74-8100（代表）</p>
        <p>窓口受付時間：午前8時30分から午後5時15分</p>
        <p>（土日、国民の祝日、年末年始12月29日から1月3日を除く）</p>
      </div>
      <div class="to-top">
        <a href="https://www.city.himi.toyama.jp/index.html">総合トップ</a>
      </div>
    </div>
  </div>
  <div class="footer-nav">
    <ul class="list">
      <li><a href="https://www.city.himi.toyama.jp/gyosei/shisei/koho_kocho/5/2597.html">個人情報の取り扱いについて</a></li>
      <li lang="en"><a href="https://www.city.himi.toyama.jp/gyosei/5429.html">Select Language</a></li>
      <li><a href="https://www.city.himi.toyama.jp/sitemap/index.html">サイトマップ</a></li>
      <li><a href="https://www.city.himi.toyama.jp/gyosei/shisei/koho_kocho/5/2598.html">免責事項について</a></li>
      <li><a href="https://www.city.himi.toyama.jp/gyosei/shisei/koho_kocho/5/2599.html">著作権について</a></li>
      <li><a href="https://www.city.himi.toyama.jp/gyosei/benri_service/shisetsu/chosha/2583.html">アクセス</a></li>
      <li><a href="https://www.city.himi.toyama.jp/gyosei/otoiawase/index.html">お問い合わせ</a></li>
    </ul>
  </div>
  <p class="copyright" lang="en">Copyright (c) 2020 Himi city. All Rights Reserved.</p>
</footer>                  </div>

      <!-- //#wrapper-in2  -->
      </div>
    <!-- //#wrapper-in  -->
    </div>
  <!-- //#wrapper  -->
  </div>

    <script src="//www.city.himi.toyama.jp/theme/base/js/external.js"></script>
        </body>
</html>`;
const doc = MDocument.fromHTML(html);

const chunks = await doc.chunk({
	headers: [
		["h1", "title"],
		["h2", "subtitle"],
	], // HTML固有のオプション
	sections: [["article", "main"]], // HTML固有のオプション
	size: 1000, // 一般的なオプション
});
console.log(chunks.length);
process.exit(0);
