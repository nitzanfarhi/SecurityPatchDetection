import csv
import itertools
import logging
import time

import collections

import requests
import os

from .create_dataset import safe_mkdir


class RepoNotFoundError(BaseException):
    pass


less_than_10_vulns = [
    '01org_opa-ff', '01org_opa-fm', '01org_tpm2.0-tools',
    '10gen-archive_mongo-c-driver-legacy', '1up-lab_oneupuploaderbundle',
    '389ds_389-ds-base', '3s3s_opentrade', '94fzb_zrlog', 'aaron-junker_usoc',
    'aaugustin_websockets', 'aawc_unrar', 'abcprintf_upload-image-with-ajax',
    'abhinavsingh_proxy.py', 'absolunet_kafe', 'acassen_keepalived',
    'accel-ppp_accel-ppp', 'accenture_mercury', 'acinq_eclair',
    'acossette_pillow', 'acpica_acpica', 'actions_http-client',
    'adaltas_node-csv-parse', 'adaltas_node-mixme', 'adamghill_django-unicorn',
    'adamhathcock_sharpcompress', 'adaptivecomputing_torque',
    'admidio_admidio', 'adodb_adodb', 'adrienverge_openfortivpn',
    'advancedforms_advanced-forms', 'afarkas_lazysizes', 'ahdinosaur_set-in',
    'aheckmann_mpath', 'aheckmann_mquery', 'aimhubio_aim', 'aio-libs_aiohttp',
    'aircrack-ng_aircrack-ng', 'airmail_airmailplugin-framework',
    'airsonic_airsonic', 'ai_nanoid', 'akashrajpurohit_clipper',
    'akheron_jansson', 'akimd_bison', 'akrennmair_newsbeuter',
    'alanaktion_phproject', 'alandekok_freeradius-server', 'alanxz_rabbitmq-c',
    'albertobeta_podcastgenerator', 'alerta_alerta', 'alexreisner_geocoder',
    'alex_rply', 'algolia_algoliasearch-helper-js', 'alkacon_apollo-template',
    'alkacon_mercury-template', 'alkacon_opencms-core', 'amazeeio_lagoon',
    'ambiot_amb1_arduino', 'ambiot_amb1_sdk', 'ampache_ampache',
    'amyers634_muracms', 'anchore_anchore-engine', 'andialbrecht_sqlparse',
    'andrerenaud_pdfgen', 'android_platform_bionic', 'andrzuk_finecms',
    'andya_cgi--simple', 'andyrixon_layerbb', 'angus-c_just',
    'ankane_chartkick', 'ansible-collections_community.crypto',
    'ansible_ansible-modules-extras', 'antonkueltz_fastecdsa',
    'antswordproject_antsword', 'anurodhp_monal', 'anymail_django-anymail',
    'aomediacodec_libavif', 'apache_activemq-artemis', 'apache_activemq',
    'apache_cordova-plugin-file-transfer',
    'apache_cordova-plugin-inappbrowser', 'apache_cxf-fediz', 'apache_cxf',
    'apache_incubator-livy', 'apache_incubator-openwhisk-runtime-docker',
    'apache_incubator-openwhisk-runtime-php', 'apache_ofbiz-framework',
    'apache_openoffice', 'apache_vcl', 'apexcharts_apexcharts.js',
    'apollosproject_apollos-apps', 'apostrophecms_apostrophe', 'apple_cups',
    'appneta_tcpreplay', 'aptana_jaxer', 'aquaverde_aquarius-core',
    'aquynh_capstone', 'arangodb_arangodb', 'archivy_archivy',
    'ardatan_graphql-tools', 'ardour_ardour', 'area17_twill', 'aresch_rencode',
    'argoproj_argo-cd', 'arjunmat_slack-chat', 'arrow-kt_arrow',
    'arsenal21_all-in-one-wordpress-security',
    'arsenal21_simple-download-monitor', 'arslancb_clipbucket',
    'artifexsoftware_ghostpdl', 'artifexsoftware_jbig2dec',
    'asaianudeep_deep-override', 'ashinn_irregex', 'askbot_askbot-devel',
    'assfugil_nickchanbot', 'asteinhauser_fat_free_crm', 'atheme_atheme',
    'atheme_charybdis', 'atinux_schema-inspector', 'att_ast',
    'auracms_auracms', 'aurelia_path', 'auth0_ad-ldap-connector',
    'auth0_auth0.js', 'auth0_express-jwt', 'auth0_express-openid-connect',
    'auth0_lock', 'auth0_nextjs-auth0', 'auth0_node-auth0',
    'auth0_node-jsonwebtoken', 'auth0_omniauth-auth0', 'authelia_authelia',
    'authguard_authguard', 'authzed_spicedb', 'automattic_genericons',
    'automattic_mongoose', 'autotrace_autotrace', 'autovance_ftp-srv',
    'avar_plack', 'avast_retdec', 'awslabs_aws-js-s3-explorer',
    'awslabs_tough', 'aws_aws-sdk-js-v3', 'aws_aws-sdk-js',
    'axdoomer_doom-vanille', 'axios_axios', 'axkibe_lsyncd', 'b-heilman_bmoor',
    'babelouest_glewlwyd', 'babelouest_ulfius', 'bacula-web_bacula-web',
    'badongdyc_fangfacms', 'bagder_curl', 'balderdashy_sails-hook-sockets',
    'ballerina-platform_ballerina-lang', 'bbangert_beaker',
    'bbengfort_confire', 'bblanchon_arduinojson', 'bblfsh_bblfshd',
    'bcfg2_bcfg2', 'bcit-ci_codeigniter', 'bcosca_fatfree-core',
    'bdew-minecraft_bdlib', 'beanshell_beanshell', 'behdad_harfbuzz',
    'belledonnecommunications_belle-sip', 'belledonnecommunications_bzrtp',
    'benjaminkott_bootstrap_package', 'bertramdev_asset-pipeline',
    'bettererrors_better_errors', 'billz_raspap-webgui', 'bit-team_backintime',
    'bitcoin_bitcoin', 'bitlbee_bitlbee', 'bitmessage_pybitmessage',
    'bittorrent_bootstrap-dht', 'blackcatdevelopment_blackcatcms',
    'blackducksoftware_hub-rest-api-python', 'blogifierdotnet_blogifier',
    'blogotext_blogotext', 'blosc_c-blosc2', 'bludit_bludit',
    'blueness_sthttpd', 'bluez_bluez', 'bminor_bash', 'bminor_glibc',
    'bonzini_qemu', 'boonebgorges_buddypress-docs', 'boonstra_slideshow',
    'boothj5_profanity', 'bottlepy_bottle', 'bouke_django-two-factor-auth',
    'bower_bower', 'boxug_trape', 'bradyvercher_gistpress',
    'braekling_wp-matomo', 'bratsche_pango', 'brave_brave-core', 'brave_muon',
    'briancappello_flask-unchained', 'brocaar_chirpstack-network-server',
    'broofa_node-uuid', 'brookinsconsulting_bccie', 'browserless_chrome',
    'browserslist_browserslist', 'browserup_browserup-proxy', 'bro_bro',
    'btcpayserver_btcpayserver', 'buddypress_buddypress',
    'bytecodealliance_lucet', 'bytom_bytom', 'c-ares_c-ares', 'c2fo_fast-csv',
    'cakephp_cakephp', 'canarymail_mailcore2', 'candlepin_candlepin',
    'candlepin_subscription-manager', 'canonicalltd_subiquity', 'caolan_forms',
    'capnproto_capnproto', 'carltongibson_django-filter',
    'carrierwaveuploader_carrierwave', 'catfan_medoo',
    'cauldrondevelopmentllc_cbang', 'ccxvii_mujs', 'cdcgov_microbetrace',
    'cdrummond_cantata', 'cdr_code-server', 'ceph_ceph-deploy',
    'ceph_ceph-iscsi-cli', 'certtools_intelmq-manager', 'cesanta_mongoose-os',
    'cesanta_mongoose', 'cesnet_perun', 'chalk_ansi-regex', 'charleskorn_kaml',
    'charybdis-ircd_charybdis', 'chaskiq_chaskiq', 'chatsecure_chatsecure-ios',
    'chatwoot_chatwoot', 'check-spelling_check-spelling', 'cherokee_webserver',
    'chevereto_chevereto-free', 'chillu_silverstripe-framework', 'chjj_marked',
    'chocolatey_boxstarter', 'chopmo_rack-ssl', 'chrisd1100_uncurl',
    'chyrp_chyrp', 'circl_ail-framework', 'cisco-talos_clamav-devel',
    'cisco_thor', 'civetweb_civetweb', 'ckeditor_ckeditor4',
    'ckolivas_cgminer', 'claviska_simple-php-captcha', 'clientio_joint',
    'cloudendpoints_esp', 'cloudfoundry_php-buildpack',
    'clusterlabs_pacemaker', 'cmuir_uncurl', 'cnlh_nps', 'cobbler_cobbler',
    'cockpit-project_cockpit', 'codecov_codecov-node',
    'codehaus-plexus_plexus-archiver', 'codehaus-plexus_plexus-utils',
    'codeigniter4_codeigniter4', 'codemirror_codemirror', 'codiad_codiad',
    'cog-creators_red-dashboard', 'cog-creators_red-discordbot',
    'collectd_collectd', 'commenthol_serialize-to-js',
    'common-workflow-language_cwlviewer', 'composer_composer',
    'composer_windows-setup', 'concrete5_concrete5-legacy',
    'containers_bubblewrap', 'containers_image', 'containers_libpod',
    'containous_traefik', 'contiki-ng_contiki-ng', 'convos-chat_convos',
    'cooltey_c.p.sub', 'coreutils_gnulib', 'corosync_corosync',
    'cosenary_instagram-php-api', 'cosmos_cosmos-sdk', 'cotonti_cotonti',
    'coturn_coturn', 'crater-invoice_crater', 'crawl_crawl',
    'creatiwity_witycms', 'creharmony_node-etsy-client',
    'crowbar_barclamp-crowbar', 'crowbar_barclamp-deployer',
    'crowbar_barclamp-trove', 'crowbar_crowbar-openstack',
    'crypto-org-chain_cronos', 'cthackers_adm-zip', 'ctripcorp_apollo',
    'ctz_rustls', 'cubecart_v6', 'cure53_dompurify', 'cvandeplas_pystemon',
    'cve-search_cve-search', 'cveproject_cvelist',
    'cyberark_conjur-oss-helm-chart', 'cyberhobo_wordpress-geo-mashup',
    'cydrobolt_polr', 'cyrusimap_cyrus-imapd', 'cyu_rack-cors',
    'd0c-s4vage_lookatme', 'd4software_querytree', 'daaku_nodejs-tmpl',
    'dagolden_capture-tiny', 'dajobe_raptor', 'daltoniam_starscream',
    'dandavison_delta', 'dankogai_p5-encode', 'danschultzer_pow',
    'darktable-org_rawspeed', 'darold_squidclamav', 'dart-lang_sdk',
    'darylldoyle_svg-sanitizer', 'dashbuilder_dashbuilder',
    'datacharmer_dbdeployer', 'datatables_datatablessrc',
    'datatables_dist-datatables', 'dav-git_dav-cogs', 'davegamble_cjson',
    'davidben_nspluginwrapper', 'davideicardi_confinit',
    'davidjclark_phpvms-popupnews', 'daylightstudio_fuel-cms',
    'dbeaver_dbeaver', 'dbijaya_onlinevotingsystem', 'dcit_perl-crypt-jwt',
    'debiki_talkyard', 'deislabs_oras', 'delta_pragyan',
    'delvedor_find-my-way', 'demon1a_discord-recon', 'denkgroot_spina',
    'deoxxa_dotty', 'dependabot_dependabot-core', 'derf_feh',
    'derickr_timelib', 'derrekr_android_security', 'desrt_systemd-shim',
    'deuxhuithuit_symphony-2', 'devsnd_cherrymusic', 'dexidp_dex',
    'dgl_cgiirc', 'dhis2_dhis2-core', 'diegohaz_bodymen', 'diegohaz_querymen',
    'dieterbe_uzbl', 'digint_btrbk', 'digitalbazaar_forge',
    'dingelish_rust-base64', 'dinhviethoa_libetpan', 'dino_dino',
    'directus_app', 'directus_directus', 'discourse_discourse-footnote',
    'discourse_discourse-reactions', 'discourse_message_bus',
    'discourse_rails_multisite', 'diversen_gallery', 'divio_django-cms',
    'diygod_rsshub', 'djabberd_djabberd', 'django-helpdesk_django-helpdesk',
    'django-wiki_django-wiki', 'dlitz_pycrypto', 'dmendel_bindata',
    'dmgerman_ninka', 'dmlc_ps-lite', 'dmproadmap_roadmap',
    'dnnsoftware_dnn.platform', 'docker_cli',
    'docker_docker-credential-helpers', 'docsifyjs_docsify', 'doctrine_dbal',
    'documize_community', 'dogtagpki_pki', 'dojo_dijit', 'dojo_dojo',
    'dojo_dojox', 'dollarshaveclub_shave', 'dom4j_dom4j', 'domoticz_domoticz',
    'dompdf_dompdf', 'doorgets_doorgets', 'doorkeeper-gem_doorkeeper',
    'dosfstools_dosfstools', 'dotcms_core', 'dotse_zonemaster-gui',
    'dottgonzo_node-promise-probe', 'dovecot_core', 'doxygen_doxygen',
    'dozermapper_dozer', 'dpgaspar_flask-appbuilder', 'dracutdevs_dracut',
    'dramforever_vscode-ghc-simple', 'drk1wi_portspoof', 'droolsjbpm_drools',
    'droolsjbpm_jbpm-designer', 'droolsjbpm_jbpm',
    'droolsjbpm_kie-wb-distributions', 'dropbox_lepton',
    'dropwizard_dropwizard', 'drudru_ansi_up', 'dspace_dspace',
    'dspinhirne_netaddr-rb', 'dsyman2_integriaims', 'dtschump_cimg',
    'duchenerc_artificial-intelligence', 'duffelhq_paginator',
    'dukereborn_cmum', 'duncaen_opendoas', 'dutchcoders_transfer.sh',
    'dvirtz_libdwarf', 'dweomer_containerd', 'dwisiswant0_apkleaks',
    'dw_mitogen', 'dynamoose_dynamoose', 'e107inc_e107',
    'e2guardian_e2guardian', 'e2openplugins_e2openplugin-openwebif',
    'eclipse-ee4j_mojarra', 'eclipse_mosquitto', 'eclipse_rdf4j',
    'eclipse_vert.x', 'edge-js_edge', 'edgexfoundry_app-functions-sdk-go',
    'edx_edx-platform', 'eflexsystems_node-samba-client', 'eggjs_extend2',
    'egroupware_egroupware', 'eiskalteschatten_compile-sass',
    'eivindfjeldstad_dot', 'elabftw_elabftw', 'elastic_elasticsearch',
    'eldy_awstats', 'elementary_switchboard-plug-bluetooth',
    'elementsproject_lightning', 'elixir-plug_plug', 'ellson_graphviz',
    'elmar_ldap-git-backup', 'elric1_knc', 'elves_elvish', 'embedthis_appweb',
    'embedthis_goahead', 'emca-it_energy-log-server-6.x', 'emlog_emlog',
    'enalean_gitphp', 'enferex_pdfresurrect', 'ensc_irssi-proxy',
    'ensdomains_ens', 'enviragallery_envira-gallery-lite', 'envoyproxy_envoy',
    'ericcornelissen_git-tag-annotation-action', 'ericcornelissen_shescape',
    'ericnorris_striptags', 'ericpaulbishop_gargoyle',
    'erikdubbelboer_phpredisadmin', 'erlang_otp', 'erlyaws_yaws',
    'esl_mongooseim', 'esnet_iperf', 'esphome_esphome', 'ethereum_go-ethereum',
    'ethereum_solidity', 'ether_ueberdb', 'ettercap_ettercap',
    'eugeneware_changeset', 'eugeny_ajenti', 'evangelion1204_multi-ini',
    'evanphx_json-patch', 'evilnet_nefarious2', 'evilpacket_marked',
    'excon_excon', 'exiftool_exiftool', 'exim_exim',
    'express-handlebars_express-handlebars', 'eyesofnetworkcommunity_eonweb',
    'ezsystems_ezjscore', 'f21_jwt', 'fabiocaccamo_utils.js', 'fabpot_twig',
    'facebookincubator_fizz', 'facebookincubator_mvfst',
    'facebookresearch_parlai', 'facebook_buck', 'facebook_folly',
    'facebook_mcrouter', 'facebook_nuclide', 'facebook_react-native',
    'facebook_wangle', 'facebook_zstd', 'faisalman_ua-parser-js',
    'faiyazalam_wordpress-plugin-user-login-history', 'fardog_trailing-slash',
    'fasterxml_jackson-dat'
]

OUTPUT_DIRNAME = 'graphql'

# copied from https://github.com/n0vad3v/get-profile-data-of-repo-stargazers-graphql

token = open(r'C:\secrets\github_token.txt', 'r').read()
headers = {"Authorization": "token " + token}

generalQL = """
{{
  repository(name: "{0}", owner: "{1}") {{
    {2}(first: 100 {3}) {{	
          totalCount
          pageInfo {{
            endCursor
            hasPreviousPage
            startCursor
          }}
          edges {{
            cursor
            node {{
              createdAt
            }}
          }}
    }}
  }}
}}

"""

stargazer_query = """
{{
  repository(name: "{0}", owner: "{1}") {{
    stargazers(first: 100 {2}) {{	
        totalCount
        pageInfo {{
        endCursor
        hasPreviousPage
        startCursor
      }}
      edges {{
        starredAt
      }}
    }}
  }}
}}
"""

# todo check if we can find commits from other branches
commits_ql = """
{{
  repository(name: "{0}",owner: "{1}") {{
    object(expression: "{2}") {{
      ... on Commit {{
        history (first:100 {3}){{
          totalCount
          pageInfo{{
            endCursor
          }}
          nodes {{
            committedDate
            deletions
            additions
            oid
          }}
          pageInfo {{
            endCursor
          }}
        }}
      }}
    }}
  }}
}}
"""

branches_ql = """
{{
  repository(owner: "{0}", name: "{1}") {{
    refs(first: 50, refPrefix:"refs/heads/") {{
      nodes {{
        name
      }}
    }}
  }}
}}

"""

repo_meta_data = """
{{
  repository(owner: "{0}", name: "{1}") {{
    owner {{
      
      ... on User {{
        company
        isEmployee
        isHireable
        isSiteAdmin
        isGitHubStar
        isSponsoringViewer
        isCampusExpert
        isDeveloperProgramMember
      }}
      ... on Organization {{
        
        isVerified        
      }}
    }}
    isInOrganization
    createdAt
    diskUsage
    hasIssuesEnabled
    hasWikiEnabled
    isMirror
    isSecurityPolicyEnabled
    fundingLinks {{
      platform
    }}
    primaryLanguage {{
      name
    }}
    languages(first: 100) {{
      edges {{
        node {{
          name
        }}
      }}
    }}
  }}
}}

"""

attrib_list = [
    "vulnerabilityAlerts", "forks", "issues", "pullRequests", "releases",
    "stargazers"
]


def run_query(query):
    """sends a query to the github graphql api and returns the result as json"""
    counter = 0
    while True:
        request = requests.post('https://api.github.com/graphql',
                                json={'query': query},
                                headers=headers)
        if request.status_code == 200:
            return request.json()
        elif request.status_code == 502:
            raise RuntimeError(
                f"Query failed to run by returning code of {request.status_code}. {request}"
            )

        else:
            request_json = request.json()
            if "errors" in request_json and (
                    "timeout" in request_json["errors"][0]["message"]
                    or request_json["errors"]["type"] == 'RATE_LIMITED'):

                print("Waiting for an hour")
                print(request, request_json)
                counter += 1
                if counter < 6:
                    time.sleep(60 * 60)
                    continue
                break

            raise RuntimeError(
                f"Query failed to run by returning code of {request.status_code}. {query}"
            )


def flatten(d, parent_key='', sep='_'):
    """flatten a nested dict"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_commit_metadata(owner, repo):
    """get commit metadata from all branches"""
    res = run_query(repo_meta_data.format(owner, repo))
    if not res['data']['repository']:
        return None
    res = flatten(res['data']['repository'])
    res['languages_edges'] = list(
        map(lambda lang: lang['node']['name'], res['languages_edges']))

    return res


def get_all_commits(owner, repo):
    """
    Get all commits from all branches
    """
    branch_lst = run_query(branches_ql.format(owner, repo))
    branch_lst = [
        res['name']
        for res in branch_lst['data']['repository']['refs']['nodes']
    ]
    commit_date, additions, deletions, oids = [], [], [], []
    final_lst = []
    if "master" in branch_lst:
        final_lst.append('master')
    if "main" in branch_lst:
        final_lst.append('main')

    for branch in branch_lst:
        print(f"\t\t{branch}")
        cur_commit_date, cur_additions, cur_deletions, cur_oids = get_commits(
            owner, repo, branch)
        commit_date += cur_commit_date
        additions += cur_additions
        deletions += cur_deletions
        oids += cur_oids
    return commit_date, additions, deletions, oids


def get_commits(owner, repo, branch):
    """Get commits from a branch"""
    endCursor = ""  # Start from begining
    this_query = commits_ql.format(repo, owner, branch, endCursor)
    commit_date, additions, deletions, oid = [], [], [], []

    result = run_query(this_query)  # Execute the query
    if "data" in result and result["data"]["repository"]["object"] is not None:
        total_count = result['data']['repository']['object']['history'][
            'totalCount']
        for _ in range(0, total_count, 100):
            endCursor = result['data']['repository']['object']['history'][
                'pageInfo']['endCursor']
            for val in result['data']['repository']['object']['history'][
                    'nodes']:
                if val is not None:
                    commit_date.append(val['committedDate'])
                    additions.append(val['additions'])
                    deletions.append(val['deletions'])
                    oid.append(val['oid'])

            result = run_query(
                commits_ql.format(repo, owner, branch,
                                  'after:"{0}"'.format(endCursor)))
            if "data" not in result:
                print("Error3", result)
                break
    else:
        print("Error4", result)

    return additions, deletions, commit_date, oid


def get_stargazers(owner, repo):
    """
    Get all the stargazers of a repo
    """
    endCursor = ""  # Start from begining
    this_query = stargazer_query.format(repo, owner, endCursor)
    has_next_page = True
    staredAt = []
    result = run_query(this_query)  # Execute the query
    if "data" in result:
        total_count = result['data']['repository']['stargazers']['totalCount']
        for _ in range(0, total_count, 100):
            endCursor = result['data']['repository']['stargazers']['pageInfo'][
                'endCursor']
            staredAt.extend(
                val['starredAt']
                for val in result['data']['repository']['stargazers']['edges'])

            result = run_query(
                stargazer_query.format(repo, owner,
                                       'after:"{0}"'.format(endCursor)))
            if "data" not in result:
                raise RuntimeError(f"result {result} does not contain data")
    else:
        logging.error(result)
        raise RuntimeError(
            f"Query failed to run by returning code of {result}. {this_query}")
    return staredAt


def get_attribute(owner, repo, attribute):
    endCursor = ""  # Start from begining
    this_query = generalQL.format(repo, owner, attribute, endCursor)
    dates = []
    result = run_query(this_query)  # Execute the query
    if 'data' in result:
        total_count = result['data']['repository'][attribute]['totalCount']
        for _ in range(0, total_count, 100):
            endCursor = result['data']['repository'][attribute]['pageInfo'][
                'endCursor']
            dates.extend(
                val['node']['createdAt']
                for val in result['data']['repository'][attribute]['edges'])

            result = run_query(
                generalQL.format(repo, owner, attribute,
                                 'after:"{0}"'.format(endCursor)))
            if 'data' not in result:
                break

    else:
        logging.error("Attribute acquire error:", result)
    return dates


def get_repo(output_dir, repo):

    safe_mkdir(os.path.join(output_dir, OUTPUT_DIRNAME))

    owner = repo.split('/')[0]
    repo = repo.split('/')[1]

    logging.debug(f"Getting repo {repo} from {owner}")
    res_dict = {}
    for attribute in attrib_list:
        logging.debug("\t" + attribute)
        if attribute == "stargazers":
            res_dict[attribute] = get_stargazers(owner, repo)
        elif attribute == "commits":
            res_dict['additions'], res_dict['deletions'], res_dict[
                'commit_date'], res_dict['oid'] = get_all_commits(owner, repo)
        else:
            res_dict[attribute] = get_attribute(owner, repo, attribute)

    with open(os.path.join(output_dir, OUTPUT_DIRNAME, f"{owner}_{repo}.csv"),
              "w",
              newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(res_dict.keys())
        writer.writerows(itertools.zip_longest(*res_dict.values()))


def get_date_for_commit(repo, commit):
    owner = repo.split('/')[0]
    repo = repo.split('/')[1]
    ql_query = """
    {{
      repository(owner: "{0}", name: "{1}") {{
        object(expression: "{2}") {{
          ... on Commit {{
            committedDate
          }}
        }}
      }}
    }}""".format(owner, repo, commit)
    result = run_query(ql_query)
    if "errors" in result:
        print("ERROR1", ql_query, result)
        raise RepoNotFoundError()
    if "data" in result and result["data"]["repository"]["object"] is not None:
        return result["data"]["repository"]["object"]["committedDate"]
    print("ERROR2", ql_query, result)
    raise RepoNotFoundError()


def get_date_for_alternate_proj_commit(proj_name, commit_hash):
    owner = proj_name.split('/')[0]
    repo = proj_name.split('/')[1]
    query = """{{
          search(query: "{0}", type: REPOSITORY, first: 100) {{
            repositoryCount
            edges {{
              node {{
                ... on Repository {{
                  nameWithOwner
                  name
                }}
              }}
            }}
          }}
        }}
    
    """

    result = run_query(query.format(repo))
    if "data" not in result:
        return None, None
    for res in result['data']['search']['edges']:
        cur_repo = res['node']['nameWithOwner']
        if res['node']['name'] != repo:
            continue
        url = "http://www.github.com/{0}/commit/{1}".format(
            cur_repo, commit_hash)
        f = requests.get(url)
        print(url, f.status_code)
        if f.status_code == 200:
            try:
                return cur_repo, get_date_for_commit(cur_repo, commit_hash)
            except RepoNotFoundError:
                pass

    return None, None


all_langs = [
    '1C Enterprise', 'AGS Script', 'AIDL', 'AMPL', 'ANTLR', 'API Blueprint',
    'ASL', 'ASP', 'ASP.NET', 'ActionScript', 'Ada', 'Agda', 'Alloy',
    'AngelScript', 'ApacheConf', 'Apex', 'AppleScript', 'Arc', 'AspectJ',
    'Assembly', 'Asymptote', 'Augeas', 'AutoHotkey', 'AutoIt', 'Awk', 'BASIC',
    'Ballerina', 'Batchfile', 'Berry', 'Bicep', 'Bikeshed', 'BitBake', 'Blade',
    'BlitzBasic', 'Boogie', 'Brainfuck', 'Brightscript', 'C', 'C#', 'C++',
    'CMake', 'COBOL', 'CSS', 'CUE', 'CWeb', 'Cadence', "Cap'n Proto", 'Ceylon',
    'Chapel', 'Charity', 'ChucK', 'Clarion', 'Classic ASP', 'Clean', 'Clojure',
    'Closure Templates', 'CodeQL', 'CoffeeScript', 'ColdFusion', 'Common Lisp',
    'Common Workflow Language', 'Coq', 'Cuda', 'Cython', 'D',
    'DIGITAL Command Language', 'DM', 'DTrace', 'Dart', 'Dhall', 'Dockerfile',
    'Dylan', 'E', 'ECL', 'EJS', 'Eiffel', 'Elixir', 'Elm', 'Emacs Lisp',
    'EmberScript', 'Erlang', 'Euphoria', 'F#', 'F*', 'FLUX', 'Fancy', 'Faust',
    'Filebench WML', 'Fluent', 'Forth', 'Fortran', 'FreeBasic', 'FreeMarker',
    'GAP', 'GCC Machine Description', 'GDB', 'GDScript', 'GLSL', 'GSC',
    'Game Maker Language', 'Genshi', 'Gherkin', 'Gnuplot', 'Go', 'Golo',
    'Gosu', 'Groff', 'Groovy', 'HCL', 'HLSL', 'HTML', 'Hack', 'Haml',
    'Handlebars', 'Haskell', 'Haxe', 'Hy', 'IDL', 'IGOR Pro', 'Inform 7',
    'Inno Setup', 'Ioke', 'Isabelle', 'Jasmin', 'Java', 'JavaScript',
    'JetBrains MPS', 'Jinja', 'Jolie', 'Jsonnet', 'Julia', 'Jupyter Notebook',
    'KRL', 'Kotlin', 'LLVM', 'LSL', 'Lasso', 'Latte', 'Less', 'Lex', 'Limbo',
    'Liquid', 'LiveScript', 'Logos', 'Lua', 'M', 'M4', 'MATLAB', 'MAXScript',
    'MLIR', 'MQL4', 'MQL5', 'Macaulay2', 'Makefile', 'Mako', 'Mathematica',
    'Max', 'Mercury', 'Meson', 'Metal', 'Modelica', 'Modula-2', 'Modula-3',
    'Module Management System', 'Monkey', 'Moocode', 'MoonScript', 'Motoko',
    'Mustache', 'NASL', 'NSIS', 'NewLisp', 'Nextflow', 'Nginx', 'Nim', 'Nit',
    'Nix', 'Nu', 'OCaml', 'Objective-C', 'Objective-C++', 'Objective-J',
    'Open Policy Agent', 'OpenEdge ABL', 'PEG.js', 'PHP', 'PLSQL', 'PLpgSQL',
    'POV-Ray SDL', 'Pan', 'Papyrus', 'Pascal', 'Pawn', 'Perl', 'Perl 6',
    'Pike', 'Pony', 'PostScript', 'PowerShell', 'Processing', 'Procfile',
    'Prolog', 'Promela', 'Pug', 'Puppet', 'PureBasic', 'PureScript', 'Python',
    'QML', 'QMake', 'R', 'RAML', 'REXX', 'RPC', 'RPGLE', 'RUNOFF', 'Racket',
    'Ragel', 'Ragel in Ruby Host', 'Raku', 'ReScript', 'Reason', 'Rebol',
    'Red', 'Redcode', 'RenderScript', 'Rich Text Format', 'Riot',
    'RobotFramework', 'Roff', 'RouterOS Script', 'Ruby', 'Rust', 'SAS', 'SCSS',
    'SMT', 'SQLPL', 'SRecode Template', 'SWIG', 'Sage', 'SaltStack', 'Sass',
    'Scala', 'Scheme', 'Scilab', 'Shell', 'ShellSession', 'Sieve', 'Slice',
    'Slim', 'SmPL', 'Smali', 'Smalltalk', 'Smarty', 'Solidity', 'SourcePawn',
    'Stan', 'Standard ML', 'Starlark', 'Stata', 'StringTemplate', 'Stylus',
    'SuperCollider', 'Svelte', 'Swift', 'SystemVerilog', 'TLA', 'TSQL', 'Tcl',
    'TeX', 'Tea', 'Terra', 'Thrift', 'Turing', 'Twig', 'TypeScript',
    'UnrealScript', 'VBA', 'VBScript', 'VCL', 'VHDL', 'Vala',
    'Velocity Template Language', 'Verilog', 'Vim Snippet', 'Vim script',
    'Visual Basic', 'Visual Basic .NET', 'Volt', 'Vue', 'WebAssembly', 'Wren',
    'X10', 'XProc', 'XQuery', 'XS', 'XSLT', 'Xtend', 'YARA', 'Yacc', 'Yul',
    'Zeek', 'Zig', 'eC', 'jq', 'kvlang', 'mupad', 'nesC', 'q', 'sed', 'xBase'
]
