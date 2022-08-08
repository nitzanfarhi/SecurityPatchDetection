from functools import wraps
from time import time
import itertools
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import array
from sklearn.metrics import precision_recall_fscore_support as f_score
from sklearn.metrics import accuracy_score as a_score
import os
import argparse
import enum


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError(
                "type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


def normalize(time_series_feature):
    if time_series_feature.max() - time_series_feature.min() == 0:
        return time_series_feature
    return (time_series_feature - time_series_feature.min()) / (time_series_feature.max() - time_series_feature.min())


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        seq_x = np.pad(seq_x, ((0, 0), (0, 30 - seq_x.shape[1])), 'constant')
        X.append(seq_x)
        y.append(seq_y[-1])
    return array(X), array(y)


def draw_timeline(name, vulns, first_date, last_date):
    dates = vulns
    dates += [first_date]
    dates += [last_date]

    values = [1] * len(dates)
    values[-1] = 2
    values[-2] = 2

    X = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.scatter(X, [1] * len(X), c=values,
               marker='s', s=100)
    fig.autofmt_xdate()

    # everything after this is turning off stuff that's plotted by default
    ax.set_title(name)
    ax.yaxis.set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_facecolor('white')

    ax.get_yaxis().set_ticklabels([])
    plt.show()


def find_best_f1(X_test, y_test, model):
    max_f1 = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test)
    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 100).astype(int)
        precision, recall, fscore, support = f_score(
            y_test, y_predict, zero_division=0)
        cur_f1 = fscore[1]
        # print(i,cur_f1)
        if cur_f1 > max_f1:
            max_f1 = cur_f1
            best_y = y_predict
            thresh = i / 100
    return max_f1, thresh, best_y


def find_best_accuracy(X_test, y_test, model):
    max_score = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test)
    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 1000).astype(int)
        score = a_score(y_test.astype(float), y_predict)
        # print(i,cur_f1)
        if score > max_score:
            max_score = score
            best_y = y_predict
            thresh = i / 100
    return max_score, thresh, best_y


def generator(feat, labels):
    pairs = [(x, y) for x in feat for y in labels]
    cycle_pairs = itertools.cycle(pairs)
    for a, b in pairs:
        yield np.array([a]), np.array([b])
    return


def find_threshold(model, x_train_scaled):
    import tensorflow as tf
    reconstructions = model.predict(x_train_scaled)
    # provides losses of individual instances
    reconstruction_errors = tf.keras.losses.msle(
        reconstructions, x_train_scaled)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) + \
        np.std(reconstruction_errors.numpy())
    return threshold


def get_predictions(model, x_test_scaled, threshold):
    import tensorflow as tf

    predictions = model.predict(x_test_scaled)
    # provides losses of individual instances
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds


token = open(r'C:\secrets\github_token.txt', 'r').read()
headers = {"Authorization": "token " + token}


commits_between_dates = """
{{
    repository(owner: "{0}", name:"{1}") {{
        object(expression: "{2}") {{
            ... on Commit {{
                history(first: 100, since: "{3}", until: "{4}") {{
                    nodes {{
                      commitUrl,
                      message
                    }}
                }}
            }}
    }}
  }}
}}




"""


def run_query(query, ignore_errors=False):
    counter = 0
    while True:
        request = requests.post(
            'https://api.github.com/graphql', json={'query': query}, headers=headers)
        if request.status_code == 200:
            return request.json()
        elif request.status_code == 502:
            raise Exception(
                "Query failed to run by returning code of {}. {}".format(request.status_code, request, query))
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

            err_string = "Query failed to run by returning code of {}. {}".format(
                request.status_code, query)
            if ignore_errors:
                print(err_string)
            else:
                raise Exception(err_string)


def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap


csvs_with_many_vulns = ['abrt_abrt',
                        'academysoftwarefoundation_openexr',
                        'agentejo_cockpit',
                        'aircrack-ng_aircrack-ng',
                        'ansible_ansible',
                        'antirez_redis',
                        'antlarr_audiofile',
                        'anuko_timetracker',
                        'apache_httpd',
                        'armmbed_mbedtls',
                        'atutor_atutor',
                        'audreyt_module-signature',
                        'axiomatic-systems_bento4',
                        'b2evolution_b2evolution',
                        'bagder_curl',
                        'bcgit_bc-java',
                        'bestpractical_rt',
                        'bfabiszewski_libmobi',
                        'bigbluebutton_bigbluebutton',
                        'bigtreecms_bigtree-cms',
                        'bottelet_daybydaycrm',
                        'cacti_cacti',
                        'cakephp_cakephp',
                        'capnproto_capnproto',
                        'ccxvii_mujs',
                        'cendioossman_tigervnc',
                        'centreon_centreon',
                        'ceph_ceph',
                        'clusterlabs_pcs',
                        'combodo_itop',
                        'craftcms_cms',
                        'danbloomberg_leptonica',
                        'daylightstudio_fuel-cms',
                        'dbry_wavpack',
                        'debiki_talkyard',
                        'discourse_discourse',
                        'django_django',
                        'djblets_djblets',
                        'dolibarr_dolibarr',
                        'dropbox_lepton',
                        'e107inc_e107',
                        'elgg_elgg',
                        'ellson_graphviz',
                        'enalean_tuleap',
                        'erikd_libsndfile',
                        'espruino_espruino',
                        'ether_etherpad-lite',
                        'exiv2_exiv2',
                        'exponentcms_exponent-cms',
                        'facebookincubator_fizz',
                        'facebook_fbthrift',
                        'facebook_folly',
                        'facebook_hermes',
                        'facebook_hhvm',
                        'facebook_proxygen',
                        'fail2ban_fail2ban',
                        'fasterxml_jackson-databind',
                        'fatfreecrm_fat_free_crm',
                        'ffmpeg_ffmpeg',
                        'file_file',
                        'flatpak_flatpak',
                        'forkcms_forkcms',
                        'freerdp_freerdp',
                        'galette_galette',
                        'getgrav_grav',
                        'git_git',
                        'glpi-project_glpi',
                        'golang_go',
                        'googlei18n_sfntly',
                        'google_asylo',
                        'gophish_gophish',
                        'gosa-project_gosa-core',
                        'gpac_gpac',
                        'handlebars-lang_handlebars.js',
                        'happyworm_jplayer',
                        'heimdal_heimdal',
                        'hoene_libmysofa',
                        'horde_horde',
                        'ifmeorg_ifme',
                        'illumos_illumos-gate',
                        'imagemagick_imagemagick',
                        'imagemagick_imagemagick6',
                        'impress-org_give',
                        'inspircd_inspircd',
                        'inverse-inc_sogo',
                        'ioquake_ioq3',
                        'iortcw_iortcw',
                        'ivmai_bdwgc',
                        'ivywe_geeklog-ivywe',
                        'janeczku_calibre-web',
                        'jedisct1_pure-ftpd',
                        'jenkinsci_jenkins',
                        'jquery_jquery-ui',
                        'jsummers_imageworsener',
                        'jupyter_notebook',
                        'kajona_kajonacms',
                        'kaltura_server',
                        'kamailio_kamailio',
                        'kanboard_kanboard',
                        'karelzak_util-linux',
                        'kde_kdeconnect-kde',
                        'kkos_oniguruma',
                        'knik0_faad2',
                        'kraih_mojo',
                        'krb5_krb5',
                        'labd_wagtail-2fa',
                        'laurent22_joplin',
                        'libarchive_libarchive',
                        'libass_libass',
                        'libav_libav',
                        'libevent_libevent',
                        'libexif_libexif',
                        'libgd_libgd',
                        'libgit2_libgit2',
                        'libguestfs_hivex',
                        'libimobiledevice_libplist',
                        'libming_libming',
                        'libraw_libraw',
                        'libredwg_libredwg',
                        'libreoffice_core',
                        'libreswan_libreswan',
                        'libvips_libvips',
                        'LibVNC_libvncserver',
                        'liferay_liferay-portal',
                        'limesurvey_limesurvey',
                        'livehelperchat_livehelperchat',
                        'lua_lua',
                        'lxc_lxc',
                        'lxml_lxml',
                        'madler_zlib',
                        'mantisbt_mantisbt',
                        'mariocasciaro_object-path',
                        'matrix-org_sydent',
                        'matroska-org_libebml',
                        'mdadams_jasper',
                        'memcached_memcached',
                        'microsoft_chakracore',
                        'microweber_microweber',
                        'mikel_mail',
                        'miniupnp_miniupnp',
                        'misp_misp',
                        'mjg59_linux',
                        'mm2_little-cms',
                        'modxcms_revolution',
                        'mono_mono',
                        'mruby_mruby',
                        'mysql_mysql-server',
                        'nagiosenterprises_nagioscore',
                        'navigatecms_navigate-cms',
                        'neomutt_neomutt',
                        'net-snmp_net-snmp',
                        'nethack_nethack',
                        'nginx_nginx',
                        'nicolargo_glances',
                        'nilsteampassnet_teampass',
                        'nodebb_nodebb',
                        'ntop_ndpi',
                        'ntop_ntopng',
                        'octobercms_library',
                        'octobercms_october',
                        'oisf_suricata',
                        'op-tee_optee_os',
                        'openbsd_src',
                        'opencast_opencast',
                        'openolat_openolat',
                        'openpgpjs_openpgpjs',
                        'opensc_opensc',
                        'openssh_openssh-portable',
                        'openssl_openssl',
                        'openstack_glance',
                        'openstack_keystone',
                        'openstack_nova',
                        'opensuse_open-build-service',
                        'opmantek_open-audit',
                        'osclass_osclass',
                        'osticket_osticket',
                        'owncloud_core',
                        'pcmacdon_jsish',
                        'perl5-dbi_dbd-mysql',
                        'perl5-dbi_dbi',
                        'pfsense_freebsd-ports',
                        'pfsense_pfsense',
                        'phoronix-test-suite_phoronix-test-suite',
                        'phpbb_phpbb',
                        'phpfusion_phpfusion',
                        'phpipam_phpipam',
                        'phpmyadmin_phpmyadmin',
                        'php_php-src',
                        'phusion_passenger',
                        'pimcore_pimcore',
                        'piwigo_piwigo',
                        'plougher_squashfs-tools',
                        'projectacrn_acrn-hypervisor',
                        'publify_publify',
                        'puppetlabs_puppet',
                        'python-pillow_pillow',
                        'python_cpython',
                        'qemu_qemu',
                        'qpdf_qpdf',
                        'quassel_quassel',
                        'qutebrowser_qutebrowser',
                        'rack_rack',
                        'radarcovid_radar-covid-android',
                        'radareorg_radare2',
                        'radare_radare2',
                        'rails_rails-html-sanitizer',
                        'rails_rails',
                        'redmine_redmine',
                        'requarks_wiki',
                        'revive-adserver_revive-adserver',
                        'roundcube_roundcubemail',
                        'rpm-software-management_rpm',
                        'rust-lang_rust',
                        's9y_serendipity',
                        'saitoha_libsixel',
                        'saltstack_salt',
                        'sandstorm-io_sandstorm',
                        'sebhildebrandt_systeminformation',
                        'selinuxproject_selinux',
                        'semplon_genixcms',
                        'serenityos_serenity',
                        'serghey-rodin_vesta',
                        'sgminer-dev_sgminer',
                        'shopware_platform',
                        'shopware_shopware',
                        'sigil-ebook_sigil',
                        'simpleledger_slpjs',
                        'sitaramc_gitolite',
                        'smartstore_smartstorenet',
                        'snipe_snipe-it',
                        'spacewalkproject_spacewalk',
                        'spiderlabs_modsecurity',
                        'sqlite_sqlite',
                        'star7th_showdoc',
                        'studio-42_elfinder',
                        'symfony_symfony',
                        'symphonycms_symphony-2',
                        'sysphonic_thetis',
                        'systemd_systemd',
                        'taglib_taglib',
                        'tats_w3m',
                        'tensorflow_tensorflow',
                        'testlinkopensourcetrms_testlink-code',
                        'the-tcpdump-group_tcpdump',
                        'theforeman_foreman',
                        'thorsten_phpmyfaq',
                        'tigervnc_tigervnc',
                        'tine20_tine-2.0-open-source-groupware-and-crm',
                        'torproject_tor',
                        'torvalds_linux',
                        'totaljs_framework',
                        'u-boot_u-boot',
                        'uclouvain_openjpeg',
                        'umbraco_umbraco-cms',
                        'unrud_radicale',
                        'unshiftio_url-parse',
                        'uriparser_uriparser',
                        'vadz_libtiff',
                        'verdammelt_tnef',
                        'videolan_vlc',
                        'vim_vim',
                        'virustotal_yara',
                        'vrtadmin_clamav-devel',
                        'web2py_web2py',
                        'webmin_webmin',
                        'weld_core',
                        'wikimedia_mediawiki',
                        'wireshark_wireshark',
                        'wolfssl_wolfssl',
                        'wordpress_wordpress-develop',
                        'wordpress_wordpress',
                        'xwiki_xwiki-platform',
                        'yeraze_ytnef',
                        'yetiforcecompany_yetiforcecrm',
                        'zammad_zammad',
                        'zmartzone_mod_auth_openidc',
                        'znc_znc',
                        'zoneminder_zoneminder',
                        'zulip_zulip']


alls = ['01org_opa-ff', '01org_opa-fm', '01org_tpm2.0-tools', '10gen-archive_mongo-c-driver-legacy', '1up-lab_oneupuploaderbundle', '389ds_389-ds-base', '3s3s_opentrade', '94fzb_zrlog', 'aaron-junker_usoc', 'aaugustin_websockets', 'aawc_unrar', 'abcprintf_upload-image-with-ajax', 'abhinavsingh_proxy.py', 'abrt_abrt', 'abrt_libreport', 'absolunet_kafe', 'academysoftwarefoundation_openexr', 'acassen_keepalived', 'accel-ppp_accel-ppp', 'accenture_mercury', 'acinq_eclair', 'acossette_pillow', 'acpica_acpica', 'actions_http-client', 'adaltas_node-csv-parse', 'adaltas_node-mixme', 'adamghill_django-unicorn', 'adamhathcock_sharpcompress', 'adaptivecomputing_torque', 'admidio_admidio', 'adodb_adodb', 'adrienverge_openfortivpn', 'advancedforms_advanced-forms', 'afarkas_lazysizes', 'agentejo_cockpit', 'ahdinosaur_set-in', 'aheckmann_mpath', 'aheckmann_mquery', 'aimhubio_aim', 'aio-libs_aiohttp', 'aircrack-ng_aircrack-ng', 'airmail_airmailplugin-framework', 'airsonic_airsonic', 'ai_nanoid', 'akashrajpurohit_clipper', 'akheron_jansson', 'akimd_bison', 'akrennmair_newsbeuter', 'alanaktion_phproject', 'alandekok_freeradius-server', 'alanxz_rabbitmq-c', 'albertobeta_podcastgenerator', 'alerta_alerta', 'alexreisner_geocoder', 'alex_rply', 'algolia_algoliasearch-helper-js', 'alkacon_apollo-template', 'alkacon_mercury-template', 'alkacon_opencms-core', 'amazeeio_lagoon', 'ambiot_amb1_arduino', 'ambiot_amb1_sdk', 'ampache_ampache', 'amyers634_muracms', 'anchore_anchore-engine', 'andialbrecht_sqlparse', 'andrerenaud_pdfgen', 'android_platform_bionic', 'andrzuk_finecms', 'andya_cgi--simple', 'andyrixon_layerbb', 'angus-c_just', 'ankane_chartkick', 'ansible-collections_community.crypto', 'ansible_ansible-modules-extras', 'ansible_ansible', 'antirez_redis', 'antlarr_audiofile', 'antonkueltz_fastecdsa', 'antswordproject_antsword', 'anuko_timetracker', 'anurodhp_monal', 'anymail_django-anymail', 'aomediacodec_libavif', 'apache_activemq-artemis', 'apache_activemq', 'apache_cordova-plugin-file-transfer', 'apache_cordova-plugin-inappbrowser', 'apache_cxf-fediz', 'apache_cxf', 'apache_httpd', 'apache_incubator-livy', 'apache_incubator-openwhisk-runtime-docker', 'apache_incubator-openwhisk-runtime-php', 'apache_ofbiz-framework', 'apache_openoffice', 'apache_vcl', 'apexcharts_apexcharts.js', 'apollosproject_apollos-apps', 'apostrophecms_apostrophe', 'apple_cups', 'appneta_tcpreplay', 'aptana_jaxer', 'aquaverde_aquarius-core', 'aquynh_capstone', 'arangodb_arangodb', 'archivy_archivy', 'ardatan_graphql-tools', 'ardour_ardour', 'area17_twill', 'aresch_rencode', 'argoproj_argo-cd', 'arjunmat_slack-chat', 'armmbed_mbedtls', 'arrow-kt_arrow', 'arsenal21_all-in-one-wordpress-security', 'arsenal21_simple-download-monitor', 'arslancb_clipbucket', 'artifexsoftware_ghostpdl', 'artifexsoftware_jbig2dec', 'asaianudeep_deep-override', 'ashinn_irregex', 'askbot_askbot-devel', 'assfugil_nickchanbot', 'asteinhauser_fat_free_crm', 'atheme_atheme', 'atheme_charybdis', 'atinux_schema-inspector', 'att_ast', 'atutor_atutor', 'audreyt_module-signature', 'auracms_auracms', 'aurelia_path', 'auth0_ad-ldap-connector', 'auth0_auth0.js', 'auth0_express-jwt', 'auth0_express-openid-connect', 'auth0_lock', 'auth0_nextjs-auth0', 'auth0_node-auth0', 'auth0_node-jsonwebtoken', 'auth0_omniauth-auth0', 'authelia_authelia', 'authguard_authguard', 'authzed_spicedb', 'automattic_genericons', 'automattic_mongoose', 'autotrace_autotrace', 'autovance_ftp-srv', 'avar_plack', 'avast_retdec', 'awslabs_aws-js-s3-explorer', 'awslabs_tough', 'aws_aws-sdk-js-v3', 'aws_aws-sdk-js', 'axdoomer_doom-vanille', 'axiomatic-systems_bento4', 'axios_axios', 'axkibe_lsyncd', 'b-heilman_bmoor', 'b2evolution_b2evolution', 'babelouest_glewlwyd', 'babelouest_ulfius', 'backstage_backstage', 'bacula-web_bacula-web', 'badongdyc_fangfacms', 'bagder_curl', 'balderdashy_sails-hook-sockets', 'ballerina-platform_ballerina-lang', 'baserproject_basercms', 'bbangert_beaker', 'bbengfort_confire', 'bblanchon_arduinojson', 'bblfsh_bblfshd', 'bcfg2_bcfg2', 'bcgit_bc-java', 'bcit-ci_codeigniter', 'bcosca_fatfree-core', 'bdew-minecraft_bdlib', 'beanshell_beanshell', 'behdad_harfbuzz', 'belledonnecommunications_belle-sip', 'belledonnecommunications_bzrtp', 'benjaminkott_bootstrap_package', 'bertramdev_asset-pipeline', 'bestpractical_rt', 'bettererrors_better_errors', 'bfabiszewski_libmobi', 'bigbluebutton_bigbluebutton', 'bigtreecms_bigtree-cms', 'billz_raspap-webgui', 'bit-team_backintime', 'bitcoin_bitcoin', 'bitlbee_bitlbee', 'bitmessage_pybitmessage', 'bittorrent_bootstrap-dht', 'blackcatdevelopment_blackcatcms', 'blackducksoftware_hub-rest-api-python', 'blogifierdotnet_blogifier', 'blogotext_blogotext', 'blosc_c-blosc2', 'bludit_bludit', 'blueness_sthttpd', 'bluez_bluez', 'bminor_bash', 'bminor_glibc', 'bolt_bolt', 'bonzini_qemu', 'bookstackapp_bookstack', 'boonebgorges_buddypress-docs', 'boonstra_slideshow', 'boothj5_profanity', 'bottelet_daybydaycrm', 'bottlepy_bottle', 'bouke_django-two-factor-auth', 'bower_bower', 'boxug_trape', 'bradyvercher_gistpress', 'braekling_wp-matomo', 'bratsche_pango', 'brave_brave-core', 'brave_muon', 'briancappello_flask-unchained', 'brocaar_chirpstack-network-server', 'broofa_node-uuid', 'brookinsconsulting_bccie', 'browserless_chrome', 'browserslist_browserslist', 'browserup_browserup-proxy', 'bro_bro', 'btcpayserver_btcpayserver', 'buddypress_buddypress', 'bytecodealliance_lucet', 'bytecodealliance_wasmtime', 'bytom_bytom', 'c-ares_c-ares', 'c2fo_fast-csv', 'cacti_cacti', 'cakephp_cakephp', 'canarymail_mailcore2', 'candlepin_candlepin', 'candlepin_subscription-manager', 'canonicalltd_subiquity', 'caolan_forms', 'capnproto_capnproto', 'carltongibson_django-filter', 'carrierwaveuploader_carrierwave', 'catfan_medoo', 'cauldrondevelopmentllc_cbang', 'ccxvii_mujs', 'cdcgov_microbetrace', 'cdrummond_cantata', 'cdr_code-server', 'cendioossman_tigervnc', 'centreon_centreon', 'ceph_ceph-deploy', 'ceph_ceph-iscsi-cli', 'ceph_ceph', 'certtools_intelmq-manager', 'cesanta_mongoose-os', 'cesanta_mongoose', 'cesnet_libyang', 'cesnet_perun', 'chalk_ansi-regex', 'chamilo_chamilo-lms', 'charleskorn_kaml', 'charybdis-ircd_charybdis', 'chaskiq_chaskiq', 'chatsecure_chatsecure-ios', 'chatwoot_chatwoot', 'check-spelling_check-spelling', 'cherokee_webserver', 'chevereto_chevereto-free', 'chillu_silverstripe-framework', 'chjj_marked', 'chocobozzz_peertube', 'chocolatey_boxstarter', 'chopmo_rack-ssl', 'chrisd1100_uncurl', 'chyrp_chyrp', 'circl_ail-framework', 'cisco-talos_clamav-devel', 'cisco_thor', 'civetweb_civetweb', 'ckeditor_ckeditor4', 'ckolivas_cgminer', 'claviska_simple-php-captcha', 'clientio_joint', 'cloudendpoints_esp', 'cloudfoundry_php-buildpack', 'clusterlabs_pacemaker', 'clusterlabs_pcs', 'cmuir_uncurl', 'cnlh_nps', 'cobbler_cobbler', 'cockpit-project_cockpit', 'codecov_codecov-node', 'codehaus-plexus_plexus-archiver', 'codehaus-plexus_plexus-utils', 'codeigniter4_codeigniter4', 'codemirror_codemirror', 'codiad_codiad', 'cog-creators_red-dashboard', 'cog-creators_red-discordbot', 'collectd_collectd', 'combodo_itop', 'commenthol_serialize-to-js', 'common-workflow-language_cwlviewer', 'composer_composer', 'composer_windows-setup', 'concrete5_concrete5-legacy', 'containerd_containerd', 'containers_bubblewrap', 'containers_image', 'containers_libpod', 'containous_traefik', 'contiki-ng_contiki-ng', 'convos-chat_convos', 'cooltey_c.p.sub', 'coreutils_gnulib', 'corosync_corosync', 'cosenary_instagram-php-api', 'cosmos_cosmos-sdk', 'cotonti_cotonti', 'coturn_coturn', 'craftcms_cms', 'crater-invoice_crater', 'crawl_crawl', 'creatiwity_witycms', 'creharmony_node-etsy-client', 'crowbar_barclamp-crowbar', 'crowbar_barclamp-deployer', 'crowbar_barclamp-trove', 'crowbar_crowbar-openstack', 'crypto-org-chain_cronos', 'cthackers_adm-zip', 'ctripcorp_apollo', 'ctz_rustls', 'cubecart_v6', 'cure53_dompurify', 'curl_curl', 'cvandeplas_pystemon', 'cve-search_cve-search', 'cveproject_cvelist', 'cyberark_conjur-oss-helm-chart', 'cyberhobo_wordpress-geo-mashup', 'cydrobolt_polr', 'cyrusimap_cyrus-imapd', 'cyu_rack-cors', 'd0c-s4vage_lookatme', 'd4software_querytree', 'daaku_nodejs-tmpl', 'dagolden_capture-tiny', 'dajobe_raptor', 'daltoniam_starscream', 'danbloomberg_leptonica', 'dandavison_delta', 'dankogai_p5-encode', 'danschultzer_pow', 'darktable-org_rawspeed', 'darold_squidclamav', 'dart-lang_sdk', 'darylldoyle_svg-sanitizer', 'dashbuilder_dashbuilder', 'datacharmer_dbdeployer', 'datatables_datatablessrc', 'datatables_dist-datatables', 'dav-git_dav-cogs', 'davegamble_cjson', 'davidben_nspluginwrapper', 'davideicardi_confinit', 'davidjclark_phpvms-popupnews', 'daylightstudio_fuel-cms', 'dbeaver_dbeaver', 'dbijaya_onlinevotingsystem', 'dbry_wavpack', 'dcit_perl-crypt-jwt', 'debiki_talkyard', 'deislabs_oras', 'delta_pragyan', 'delvedor_find-my-way', 'demon1a_discord-recon', 'denkgroot_spina', 'deoxxa_dotty', 'dependabot_dependabot-core', 'derf_feh', 'derickr_timelib', 'derrekr_android_security', 'desrt_systemd-shim', 'deuxhuithuit_symphony-2', 'devsnd_cherrymusic', 'dexidp_dex', 'dgl_cgiirc', 'dhis2_dhis2-core', 'diegohaz_bodymen', 'diegohaz_querymen', 'dieterbe_uzbl', 'digint_btrbk', 'digitalbazaar_forge', 'dingelish_rust-base64', 'dinhviethoa_libetpan', 'dino_dino', 'directus_app', 'directus_directus', 'discourse_discourse-footnote', 'discourse_discourse-reactions', 'discourse_discourse', 'discourse_message_bus', 'discourse_rails_multisite', 'diversen_gallery', 'divio_django-cms', 'diygod_rsshub', 'djabberd_djabberd', 'django-helpdesk_django-helpdesk', 'django-wiki_django-wiki', 'django_django', 'djblets_djblets', 'dlitz_pycrypto', 'dmendel_bindata', 'dmgerman_ninka', 'dmlc_ps-lite', 'dmproadmap_roadmap', 'dnnsoftware_dnn.platform', 'docker_cli', 'docker_docker-credential-helpers', 'docsifyjs_docsify', 'doctrine_dbal', 'documize_community', 'dogtagpki_pki', 'dojo_dijit', 'dojo_dojo', 'dojo_dojox',
 'dolibarr_dolibarr', 'dollarshaveclub_shave', 'dom4j_dom4j', 'domoticz_domoticz', 'dompdf_dompdf', 
 'doorgets_doorgets', 'doorkeeper-gem_doorkeeper', 'dosfstools_dosfstools', 'dotcms_core', 'dotse_zonemaster-gui', 'dottgonzo_node-promise-probe', 'dovecot_core', 'doxygen_doxygen', 'dozermapper_dozer', 'dpgaspar_flask-appbuilder', 'dracutdevs_dracut', 'dramforever_vscode-ghc-simple', 'drk1wi_portspoof', 'droolsjbpm_drools', 'droolsjbpm_jbpm-designer', 'droolsjbpm_jbpm', 'droolsjbpm_kie-wb-distributions', 'dropbox_lepton', 'dropwizard_dropwizard', 'drudru_ansi_up', 'dspace_dspace', 'dspinhirne_netaddr-rb', 'dsyman2_integriaims', 'dtschump_cimg', 'duchenerc_artificial-intelligence', 'duffelhq_paginator', 'dukereborn_cmum', 'duncaen_opendoas', 'dutchcoders_transfer.sh', 'dvirtz_libdwarf', 'dweomer_containerd', 'dwisiswant0_apkleaks', 'dw_mitogen', 'dynamoose_dynamoose', 'e107inc_e107', 'e2guardian_e2guardian', 'e2openplugins_e2openplugin-openwebif', 'eclipse-ee4j_mojarra', 'eclipse_mosquitto', 'eclipse_rdf4j', 'eclipse_vert.x', 'edge-js_edge', 'edgexfoundry_app-functions-sdk-go', 'edx_edx-platform', 'eflexsystems_node-samba-client', 'eggjs_extend2', 'egroupware_egroupware', 'eiskalteschatten_compile-sass', 'eivindfjeldstad_dot', 'elabftw_elabftw', 'elastic_elasticsearch', 'eldy_awstats', 'electron_electron', 'elementary_switchboard-plug-bluetooth', 'elementsproject_lightning', 'elgg_elgg', 'elixir-plug_plug', 'ellson_graphviz', 'elmar_ldap-git-backup', 'elric1_knc', 'elves_elvish', 'embedthis_appweb', 'embedthis_goahead', 'emca-it_energy-log-server-6.x', 'emlog_emlog', 'enalean_gitphp', 'enalean_tuleap', 'enferex_pdfresurrect', 'ensc_irssi-proxy', 'ensdomains_ens', 'enviragallery_envira-gallery-lite', 'envoyproxy_envoy', 'ericcornelissen_git-tag-annotation-action', 'ericcornelissen_shescape', 'ericnorris_striptags', 'ericpaulbishop_gargoyle', 'erikdubbelboer_phpredisadmin', 'erikd_libsndfile', 'erlang_otp', 'erlyaws_yaws', 'esl_mongooseim', 'esnet_iperf', 'esphome_esphome', 'espocrm_espocrm', 'espruino_espruino', 'ethereum_go-ethereum', 'ethereum_solidity', 'ether_etherpad-lite', 'ether_ueberdb', 'ettercap_ettercap', 'eugeneware_changeset', 'eugeny_ajenti', 'evangelion1204_multi-ini', 'evanphx_json-patch', 'evilnet_nefarious2', 'evilpacket_marked', 'excon_excon', 'exiftool_exiftool', 'exim_exim', 'exiv2_exiv2', 'exponentcms_exponent-cms', 'express-handlebars_express-handlebars', 'eyesofnetworkcommunity_eonweb', 'ezsystems_ezjscore', 'f21_jwt', 'fabiocaccamo_utils.js', 'fabpot_twig', 'facebookincubator_fizz', 'facebookincubator_mvfst', 'facebookresearch_parlai', 'facebook_buck', 'facebook_fbthrift', 'facebook_folly', 'facebook_hermes', 'facebook_hhvm', 'facebook_mcrouter', 'facebook_nuclide', 'facebook_proxygen', 'facebook_react-native', 'facebook_wangle', 'facebook_zstd', 'fail2ban_fail2ban', 'faisalman_ua-parser-js', 'faiyazalam_wordpress-plugin-user-login-history', 'fardog_trailing-slash', 'fasterxml_jackson-databind', 'fasterxml_jackson-dataformats-binary', 'fastify_fastify-http-proxy', 'fastify_fastify-reply-from', 'fastspot_bigtree-form-builder', 'fatcerberus_minisphere', 'fatfreecrm_fat_free_crm', 'faye_faye', 'faye_websocket-extensions-node', 'faye_websocket-extensions-ruby', 'fb55_nth-check', 'fbb-git_yodl', 'fecshop_yii2_fecshop', 'federicoceratto_nim-httpauth', 'fedora-infra_mirrormanager2', 'fedora-infra_python-fedora', 'fedora-selinux_selinux-policy', 'fedora-selinux_setroubleshoot', 'felixrieseberg_windows-build-tools', 'ffay_lanproxy', 'ffi_ffi', 'ffmpeg_ffmpeg', 'fgasper_p5-crypt-perl', 'filebrowser_filebrowser', 'file_file', 'filp_whoops', 'firebase_firebase-js-sdk', 'fireblinkltd_object-collider', 'firefly-iii_firefly-iii', 'firelyteam_spark', 'fish-shell_fish-shell', 'fiveai_cachet', 'fiznool_body-parser-xml', 'flarum_core', 'flarum_sticky', 'flask-middleware_flask-security', 'flatcore_flatcore-cms', 'flatpak_flatpak', 'fleetdm_fleet', 'flexpaper_pdf2json', 'flexsolution_alfrescoresetpassword', 'flitbit_json-ptr', 'flori_json', 'flowplayer_flowplayer', 'fluent-plugins-nursery_td-agent-builder', 'fluent_fluent-bit', 'flyspray_flyspray', 'fmtlib_fmt', 'follow-redirects_follow-redirects', 'fontforge_fontforge', 'fordnn_usersexportimport', 'forkcms_forkcms', 'formspree_formspree', 'fosnola_libstaroffice', 'fracpete_vfsjfilechooser2', 'fragglet_lhasa', 'frankenderman_butter', 'fransurbo_zfs', 'freebsd_freebsd-src', 'freepbx_contactmanager', 'freepbx_fw_ari', 'freepbx_manager', 'freeradius_freeradius-server', 'freeradius_pam_radius', 'freerdp_freerdp', 'freertos_freertos-kernel', 'fribidi_fribidi', 'friendica_friendica', 'friendsoftypo3_mediace', 'froxlor_froxlor', 'frrouting_frr', 'funzoneq_freshdns', 'fusesource_hawtjni', 'fusionauth_fusionauth-samlv2', 'fusioninventory_fusioninventory-for-glpi', 'fusionpbx_fusionpbx', 'futurepress_epub.js', 'fuzzard_xbmc', 'galette_galette', 'gallery_gallery3', 'gamonoid_icehrm', 'ganglia_ganglia-web', 'gazay_gon', 'gburton_ce-phoenix', 'gburton_oscommerce2', 'gchq_cyberchef', 'ge0rg_yaxim', 'geddy_geddy', 'geminabox_geminabox', 'gemini-testing_png-img', 'gemorroj_phpwhois', 'genymobile_f2ut_platform_frameworks_base', 'gerapy_gerapy', 'getgrav_grav-plugin-admin', 'getgrav_grav', 'getk2_k2', 'getkirby-v2_panel', 'getkirby_kirby', 'getpatchwork_patchwork', 'getredash_redash', 'getsentry_raven-ruby', 'getsimplecms_getsimplecms', 'gettalong_kramdown', 'geysermc_geyser', 'ghantoos_lshell', 'gimly_vscode-matlab', 'gisle_html-parser', 'git-lfs_git-lfs', 'github_cmark-gfm', 'github_codeql-action', 'github_hub', 'github_hubot-scripts', 'github_paste-markdown', 'gitpod-io_gitpod', 'git_git', 'glennrp_libpng', 'glensc_file', 'glpi-project_glpi', 'gnachman_iterm2', 'gnome-exe-thumbnailer_gnome-exe-thumbnailer', 'gnome_evince', 'gnome_gimp', 'gnome_gnome-session', 'gnome_gnome-shell', 'gnome_libgsf', 'gnome_librsvg', 'gnome_libxml2', 'gnome_nautilus', 'gnome_pango', 'gns3_ubridge', 'gnuaspell_aspell', 'gnuboard_gnuboard5', 'go-ldap_ldap', 'go-vela_compiler', 'go-vela_server', 'gobby_libinfinity', 'gocd_gocd', 'godaddy_node-config-shield', 'godotengine_godot', 'gofiber_fiber', 'gogits_gogs', 'gogo_protobuf', 'gogs_gogs', 'goharbor_harbor', 'golang_crypto', 'golang_gddo', 'golang_go', 'golang_net', 'gollum_gollum', 'gollum_grit_adapter', 'gonicus_gosa', 'googleapis_google-oauth-java-client', 'googlechrome_rendertron', 'googlei18n_sfntly', 'google_asylo', 'google_closure-library', 'google_exposure-notifications-internals', 'google_exposure-notifications-verification-server', 'google_fscrypt', 'google_guava', 'google_gvisor', 'google_oss-fuzz-vulns', 'google_tink', 'google_voice-builder', 'gophish_gophish', 'gopro_gpmf-parser', 'gosa-project_gosa-core', 'gpac_gpac', 'gpcsolutions_dolibarr', 'gpg_boa', 'gpg_libgcrypt', 'gradio-app_gradio', 'gradle_gradle', 'grafana_agent', 'grafana_grafana', 'grandt_phprelativepath', 'graphhopper_graphhopper', 'graphpaperpress_sell-media', 'graphql_graphiql', 'graphql_graphql-playground', 'grassrootza_grassroot-platform', 'grocy_grocy', 'gruntjs_grunt', 'gsliepen_tinc', 'gssapi_gssproxy', 'gstreamer_gst-plugins-ugly', 'guymograbi_kill-by-port', 'gzpan123_pip', 'h2database_h2database', 'h2o_h2o', 'haakonnessjoen_mac-telnet', 'haf_dotnetzip.semverd', 'halostatue_minitar', 'haml_haml', 'handlebars-lang_handlebars.js', 'handsontable_formula-parser', 'hapijs_bassmaster', 'hapijs_crumb', 'hapijs_hapi', 'hapijs_hoek', 'hapijs_inert', 'hapijs_nes', 'happyworm_jplayer', 'haproxy_haproxy', 'harfbuzz_harfbuzz', 'hasura_graphql-engine', 'hawtio_hawtio', 'hedgedoc_hedgedoc', 'heimdal_heimdal', 'helm_helm', 'helpyio_helpy', 'henrikjoreteg_html-parse-stringify', 'hercules-team_augeas', 'hestiacp_hestiacp', 'hewlettpackard_nagios-plugins-hpilo', 'hexchat_hexchat', 'hexojs_hexo', 'hfp_libxsmm', 'highcharts_highcharts', 'highlightjs_highlight.js', 'hlfshell_controlled-merge', 'hoene_libmysofa', 'honzabilek4_app', 'hoppscotch_hoppscotch', 'horazont_xmpp-http-upload', 'horde_base', 'horde_gollem', 'horde_horde', 'horde_kronolith', 'houseabsolute_data-validate-ip', 'hpcng_singularity', 'hs-web_hsweb-framework', 'hsimpson_vscode-glsllint', 'html5lib_html5lib-python', 'http4s_blaze', 'http4s_http4s', 'httplib2_httplib2', 'hubspot_jinjava', 'hueniverse_hawk', 'humhub_humhub', 'hyperium_hyper', 'hyperledger_besu', 'hyperledger_indy-node', 'i7media_mojoportal', 'ibmdb_node-ibm_db', 'ibus_ibus-anthy', 'icecoder_icecoder', 'icewind1991_smb', 'icinga_icinga2', 'identitypython_pysaml2', 'identityserver_identityserver4', 'ifmeorg_ifme', 'ignacionelson_projectsend', 'igniterealtime_openfire', 'igniterealtime_smack', 'igrr_axtls-8266', 'ikeay_vinemv', 'ilanschnell_bsdiff4', 'ilchcms_ilch-2.0', 'ilias-elearning_ilias', 'illumos_illumos-gate', 'illydth_wowraidmanager', 'imagemagick_imagemagick', 'imagemagick_imagemagick6', 'imagemagick_librsvg', 'immerjs_immer', 'impress-org_give', 'impulseadventure_jpegsnoop', 'in-toto_in-toto-golang', 'indutny_elliptic', 'inextrix_astpp', 'infinispan_infinispan', 'influxdata_influxdb', 'inliniac_suricata', 'inspircd_inspircd', 'instantupdate_cms', 'intelliants_subrion', 'internationalscratchwiki_mediawiki-scratch-login', 'internationalscratchwiki_wiki-scratchsig', 'intranda_goobi-viewer-core', 'inunosinsi_soycms', 'inveniosoftware_invenio-drafts-resources', 'inverse-inc_sogo', 'invoiceninja_invoiceninja', 'invoiceplane_invoiceplane', 'iobroker_iobroker.admin', 'iobroker_iobroker.js-controller', 'ionic-team_cordova-plugin-ios-keychain', 'ionicabizau_set-or-get.js', 'ioquake_ioq3', 'iortcw_iortcw', 'ipfire_ipfire-2.x', 'ipfs_go-ipfs', 'ipmitool_ipmitool', 'ipython_ipython', 'irmen_pyro3', 'irrelon_irrelon-path', 'irssi_irssi', 'irssi_scripts.irssi.org', 'istio_envoy', 'istio_istio', 'isucon_isucon5-qualify', 'it-novum_openitcockpit', 'its-a-feature_apfell', 'iubenda_iubenda-cookie-class', 'ivmai_bdwgc', 'ivywe_geeklog-ivywe', 'jabberd2_jabberd2', 'jackalope_jackalope-doctrine-dbal', 'jacoders_openjk', 'jamesagnew_hapi-fhir', 'jamesheinrich_phpthumb', 'janeczku_calibre-web', 'janl_node-jsonpointer', 'jappix_jappix', 'jaredhanson_oauth2orize-fprm', 'jaredhanson_passport-oauth2', 'jaredly_hexo-admin', 'jarofghosts_glance', 'jasig_cas', 'jasig_dotnet-cas-client', 'jasig_java-cas-client', 'jasongoodwin_authentikat-jwt', 'jasper-software_jasper', 'javaee_mojarra', 'javamelody_javamelody', 'javaserverfaces_mojarra', 'jaw187_node-traceroute', 'jazzband_django-user-sessions', 'jborg_attic', 'jboss-fuse_fuse', 'jbroadway_elefant', 'jcampbell1_simple-file-manager', 'jcbrand_converse.js', 'jcubic_jquery.terminal', 'jcupitt_libvips', 'jdennis_keycloak-httpd-client-install', 'jdhwpgmbca_pcapture', 'jech_polipo', 'jedisct1_pure-ftpd', 'jellyfin_jellyfin', 'jenkinsci_jenkins', 'jenkinsci_subversion-plugin', 'jenkinsci_winstone', 'jeromedevome_grr', 'jerryscript-project_jerryscript', 'jessie-codes_safe-flat', 'jfhbrook_node-ecstatic', 'jgm_gitit', 'jgraph_mxgraph', 'jhipster_generator-jhipster', 'jhipster_jhipster-kotlin', 'jhuckaby_pixl-class', 'jitsi_jitsi-meet-electron', 'jitsi_jitsi', 'jktjkt_trojita', 'jmacd_xdelta-devel', 'jmrozanec_cron-utils', 'jnbt_vscode-rufo', 'jnunemaker_crack', 'jnunemaker_httparty', 'joeattardi_emoji-button', 'joescho_get-ip-range', 'johndyer_mediaelement', 'jojocms_jojo-cms', 'joniles_mpxj', 'jonrohan_zeroclipboard', 'jonschlinkert_assign-deep', 'jonschlinkert_defaults-deep', 'jonschlinkert_merge-deep', 'jonschlinkert_mixin-deep', 'jonschlinkert_set-value', 'jooby-project_jooby', 'joomla_joomla-cms', 'josdejong_jsoneditor', 'josdejong_mathjs', 'josdejong_typed-function', 'josephernest_void', 'joshf_burden', 'josh_rack-ssl', 'joyent_node', 'jpirko_libndp', 'jpuri_react-draft-wysiwyg', 'jquense_expr', 'jquery-validation_jquery-validation', 'jquery_jquery-ui', 'jquery_jquery', 'jshmrtn_hygeia', 'jsomara_katello', 'json-c_json-c', 'jsreport_jsreport-chrome-pdf', 'jsuites_jsuites', 'jsummers_deark', 'jsummers_imageworsener', 'jtdowney_private_address_check', 'juliangruber_brace-expansion', 'julianlam_nodebb-plugin-markdown', 'junit-team_junit4', 'junrar_junrar', 'jupyter-server_jupyter_server', 'jupyterhub_binderhub', 'jupyterhub_jupyterhub', 'jupyterhub_kubespawner', 'jupyterhub_nbgitpuller', 'jupyterhub_oauthenticator', 'jupyterhub_systemdspawner', 'jupyterlab_jupyterlab', 'jupyter_nbdime', 'jupyter_notebook', 'justarchinet_archisteamfarm', 'justdan96_tsmuxer', 'justingit_dada-mail', 'k-takata_onigmo', 'kajona_kajonacms', 'kaltura_server', 'kamadak_exif-rs', 'kamailio_kamailio', 'kaminari_kaminari', 'kanaka_novnc', 'kanboard_kanboard', 'karelzak_util-linux', 'karsonzhang_fastadmin', 'kataras_iris', 'katello_katello-installer', 'katello_katello', 'kawasima_struts1-forever', 'kde_ark', 'kde_kde1-kdebase', 'kde_kdeconnect-kde', 'keepkey_keepkey-firmware', 'keithw_mosh', 'kellyselden_git-diff-apply', 'kennethreitz_requests', 'kenny2github_report', 'kerolasa_lelux-utiliteetit', 'keszybz_systemd', 'kevinpapst_kimai2', 'keycloak_keycloak-documentation', 'keycloak_keycloak', 'keystonejs_keystone', 'khaledhosny_ots', 'kiegroup_jbpm-designer', 'kiegroup_jbpm-wb', 'kirilkirkov_ecommerce-codeigniter-bootstrap', 'kitabisa_teler', 'kizniche_mycodo', 'kkos_oniguruma', 'klacke_yaws', 'klaussilveira_gitlist', 'kmackay_micro-ecc', 'kmatheussen_das_watchdog', 'kn007_silk-v3-decoder', 'knik0_faad2', 'knowledgecode_date-and-time', 'koala-framework_koala-framework', 'kobebeauty_php-contact-form', 'kohler_gifsicle', 'kohler_t1utils', 'kolya5544_bearftp', 'kongchuanhujiao_server', 'kong_docker-kong', 'kong_docs.konghq.com', 'konloch_bytecode-viewer', 'koral--_android-gif-drawable', 'kovidgoyal_calibre', 'kovidgoyal_kitty', 'kozea_cairosvg', 'kozea_radicale', 'kraih_mojo', 'kravietz_pam_tacplus', 'krb5_krb5', 'kriszyp_json-schema', 'kr_beanstalkd', 'kucherenko_blamer', 'kylebrowning_apnswift', 'kyz_libmspack', 'kzar_watchadblock', 'labd_wagtail-2fa', 'labocnil_cookieviz', 'laminas_laminas-http', 'laravel-backpack_crud', 'laravel_framework', 'latchset_jwcrypto', 'latchset_kdcproxy', 'laurent22_joplin', 'laurenttreguier_vscode-rpm-spec', 'laverdet_isolated-vm', 'lavv17_lftp', 'lawngnome_php-radius', 'lcobucci_jwt', 'leanote_desktop-app', 'leantime_leantime', 'ledgersmb_ledgersmb', 'leenooks_phpldapadmin', 'legrandin_pycrypto', 'leizongmin_tomato', 'lepture_mistune', 'lesterchan_wp-dbmanager', 'libarchive_libarchive', 'libass_libass', 'libav_libav', 'libevent_libevent', 'libexif_exif', 'libexif_libexif', 'libexpat_libexpat', 'libgd_libgd', 'libgit2_libgit2', 'libguestfs_hivex', 'libguestfs_libguestfs', 'libidn_libidn2', 'libimobiledevice_libimobiledevice', 'libimobiledevice_libplist', 'libimobiledevice_libusbmuxd', 'libjpeg-turbo_libjpeg-turbo', 'libjxl_libjxl', 'liblime_liblime-koha', 'liblouis_liblouis', 'libming_libming', 'libofx_libofx', 'libraw_libraw-demosaic-pack-gpl2', 'libraw_libraw', 'libra_libra', 'libredwg_libredwg', 'librenms_librenms', 'libreoffice_core', 'libressl-portable_openbsd', 'libressl-portable_portable', 'libreswan_libreswan', 'librit_passhport', 'libssh2_libssh2', 'libtom_libtomcrypt', 'libuv_libuv', 'libvips_libvips', 'libvirt_libvirt', 'LibVNC_libvncserver', 'libvnc_x11vnc', 'libyal_libevt', 'liferay_liferay-portal', 'lift_framework', 'lightningnetwork_lnd', 'lightsaml_lightsaml', 'lighttpd_lighttpd1.4', 'limesurvey_limesurvey', 'linbit_csync2', 'line_armeria', 'lingej_pnp4nagios', 'linlinjava_litemall', 'linux4sam_at91bootstrap', 'linuxdeepin_deepin-clone', 'litecart_litecart', 'livehelperchat_livehelperchat', 'lkiesow_python-feedgen', 'llk_scratch-svg-renderer', 'locutusjs_locutus', 'locutusofborg_ettercap', 'lodash_lodash', 'log4js-node_log4js-node', 'loicmarechal_libmeshb', 'loklak_loklak_server', 'loomio_loomio', 'lora-net_loramac-node', 'lostintangent_gistpad', 'lsegal_yard', 'lua_lua', 'lucee_lucee', 'lukashinsch_spring-boot-actuator-logview', 'luke-jr_bfgminer', 'lukeed_tempura', 'lurcher_unixodbc', 'lxc_lxc', 'lxc_lxcfs', 'lxc_lxd', 'lxml_lxml', 'm6w6_ext-http', 'macournoyer_thin', 'madler_pigz', 'madler_zlib', 'maekitalo_tntnet', 'mafintosh_dns-packet', 'mafintosh_is-my-json-valid', 'mafintosh_tar-fs', 'mailcleaner_mailcleaner', 'mailcow_mailcow-dockerized', 'mailpile_mailpile', 'mandatoryprogrammer_xsshunter-express', 'mantisbt-plugins_source-integration',
    'mantisbt_mantisbt', 'manvel-khnkoyan_jpv', 'manydesigns_portofino', 'mapfish_mapfish-print',
    'mapserver_mapserver', 'marak_colors.js', 'mardiros_pyshop', 'mariadb-corporation_mariadb-connector-c',
    'mariadb_server', 'mariocasciaro_object-path', 'markdown-it_markdown-it', 'markedjs_marked', 'markevans_dragonfly',
    'martinjw_dbschemareader', 'martinpitt_python-dbusmock', 'mastercactapus_proxyprotocol', 'mathjax_mathjax', 'matrix-org_matrix-appservice-bridge', 
    'matrix-org_matrix-react-sdk', 'matrix-org_sydent', 'matrix-org_synapse', 'matroska-org_libebml', 'matroska-org_libmatroska', 'mattermost_focalboard',
    'mattermost_mattermost-server', 'mattinsler_connie-lang', 'maxsite_cms', 'mayan-edms_mayan-edms', 'mayankmetha_rucky', 'mcollina_msgpack5', 
    'mdadams_jasper', 'mdbtools_mdbtools', 'mde_ejs', 'medialize_uri.js', 'meetecho_janus-gateway', 'memcached_memcached', 'mercurius-js_mercurius',
    'mermaid-js_mermaid', 'metabase_metabase', 'mganss_htmlsanitizer', 'mguinness_elfinder.aspnet', 'mholt_archiver', 'mibew_mibew', 
    'michaelaquilina_zsh-autoswitch-virtualenv', 'michaelforney_samurai', 'michaelrsweet_htmldoc', 'michaelryanmcneill_shibboleth', 'michaelschwarz_ajax.net-professional', 'micromata_projectforge-webapp', 'micronaut-projects_micronaut-core', 'microsoft_chakracore', 'microsoft_git-credential-manager-core', 'microsoft_onefuzz', 'microweber_microweber', 'mikaelbr_mversion', 'mikaku_monitorix', 'mikel_mail', 'milkytracker_milkytracker', 'miltonio_milton2', 'mindwerks_wildmidi', 'minerstat_minerstat-os', 'minimagick_minimagick', 'minio_minio', 'miniprofiler_rack-mini-profiler', 'miniupnp_miniupnp', 'miniupnp_ngiflib', 'mintty_mintty', 'mirahezebots_sopel-channelmgnt', 'miraheze_datadump', 'miraheze_globalnewfiles', 'miraheze_managewiki', 'mirumee_saleor-storefront', 'mirumee_saleor', 'misp_misp-maltego', 'misp_misp', 'misskey-dev_misskey', 'mithunsatheesh_node-rules', 'mitreid-connect_openid-connect-java-spring-server', 'mitsuhiko_jinja2', 'mity_md4c', 'mjg59_linux', 'mjg59_pupnp-code', 'mjmlio_mjml', 'mjpclab_object-hierarchy-access', 'mjurczak_mbed-coap', 'mjwwit_node-xmlhttprequest', 'mkdynamic_omniauth-facebook', 'mkj_dropbear', 'mltframework_shotcut', 'mm2_little-cms', 'mnelson4_printmyblog', 'mnkras_concrete5', 'mnoorenberghe_zoneminder', 'moby_moby', 'mochi_mochiweb', 'modxcms_revolution', 'moinwiki_moin-1.9', 'monero-project_monero-gui', 'monetra_mstdlib', 'mongo-express_mongo-express', 'mongodb_bson-ruby', 'mongodb_js-bson', 'mongodb_mongo-c-driver', 'mongodb_mongo-python-driver', 'mongodb_mongo', 'mongoid_moped', 'monkey_monkey', 'mono_mono', 'monstra-cms_monstra', 'moodle_moodle', 'mooltipass_moolticute', 'moonlight-stream_moonlight-ios', 'morethanwords_tweb', 'morgan-phoenix_enrocrypt', 'movim_moxl', 'moxiecode_moxieplayer', 'moxiecode_plupload', 'mozilla-b2g_gaia', 'mozilla_bleach', 'mozilla_pollbot', 'mpdavis_python-jose', 'mpetroff_pannellum', 'mpv-player_mpv', 'mqttjs_mqtt.js', 'mrash_fwsnort', 'mrdoob_three.js', 'mrodrig_doc-path', 'mrrio_jspdf', 'mrswitch_hello.js', 'mruby_mruby', 'mrvautin_expresscart', 'mscdex_ssh2', 'mtrmac_iptables-parse', 'mtrojnar_stunnel', 'mumble-voip_mumble', 'munin-monitoring_munin', 'muttmua_mutt', 'mybb_mybb', 'mycolorway_simditor', 'myigel_engelsystem', 'mymarilyn_clickhouse-driver', 'myshenin_aws-lambda-multipart-parser', 'mysql_mysql-server', 'mytrile_node-libnotify', 'myvesta_vesta', 'mz-automation_libiec61850', 'nacl-ltd_pref-shimane-cms', 'nagiosenterprises_nagioscore', 'nagvis_nagvis', 'nanopb_nanopb', 'nationalsecurityagency_emissary', 'nationalsecurityagency_ghidra', 'nats-io_nats-server', 'nats-io_nats.ws', 'navigatecms_navigate-cms', 'nayutaco_ptarmigan', 'ned14_nedmalloc', 'neomutt_neomutt', 'neoraider_fastd', 'neos_form', 'neovim_neovim', 'nervjs_taro', 'net-snmp_net-snmp', 'netblue30_firejail', 'netdata_netdata', 'netdisco_netdisco', 'netflix_security_monkey', 'nethack_nethack', 'netless-io_flat-server', 'netristv_ws-scrcpy', 'nette_latte', 'netty_netty', 'neuecc_messagepack-csharp', 'neutrinolabs_xrdp', 'nextcloud_android', 'nextcloud_apps', 'nextcloud_circles', 'nextcloud_gallery', 'nextcloud_news-android', 'nextcloud_server', 'nfriedly_node-bestzip', 'ng-packagr_ng-packagr', 'nghttp2_nghttp2', 'nginx_nginx', 'nhosoya_omniauth-apple', 'nicolargo_glances', 'nicolas-van_modern-async', 'nightscout_cgm-remote-monitor', 'nih-at_libzip', 'niklasmerz_cordova-plugin-fingerprint-aio', 'nilsteampassnet_teampass', 'nim-lang_nimble', 'nimble-platform_common', 'nitrokey_nitrokey-fido-u2f-firmware', 'nixos_nixpkgs', 'nltk_nltk', 'nmap_nmap', 'nocodb_nocodb', 'node-fetch_node-fetch', 'node-red_node-red', 'nodebb_nodebb', 'nodejs_node', 'nodemailer_nodemailer', 'noderedis_node-redis', 'nolimits4web_swiper', 'nordaaker_convos', 'nordicsemiconductor_android-ble-library', 'nordicsemiconductor_android-dfu-library', 'not-kennethreitz_envoy', 'nothings_stb', 'novnc_novnc', 'nov_jose-php', 'nov_json-jwt', 'nozbe_watermelondb', 'npat-efault_picocom', 'npm_cli', 'npm_fstream', 'npm_hosted-git-info', 'npm_ini', 'npm_node-tar', 'npm_npm-user-validate', 'npm_npm', 'nshahzad_phpvms', 'ntop_ndpi', 'ntop_ntopng', 'ntp-project_ntp', 'nukeviet_module-shops', 'nukeviet_nukeviet', 'numpy_numpy', 'nuxeo_richfaces', 'nuxsmin_syspass', 'nuxt_nuxt.js', 'nvbn_thefuck', 'nystudio107_craft-seomatic', 'oauth2-proxy_oauth2-proxy', 'objsys_oocborrt', 'oblac_jodd', 'ocaml_ocaml', 'ocsinventory-ng_ocsinventory-ocsreports', 'octobercms_library', 'octobercms_october', 'octopusdeploy_octopusdsc', 'odoo_odoo', 'oetiker_rrdtool-1.x', 'oetiker_smokeping', 'ohler55_agoo', 'ohmyzsh_ohmyzsh', 'oisf_suricata', 'okws_okws', 'oliversalzburg_i18n-node-angular', 'omeka_omeka', 'ome_omero-web', 'omniauth_omniauth', 'omphalos_crud-file-server', 'omrilotan_async-git', 'onelogin_wordpress-saml', 'onlyoffice_plugin-translator', 'op-tee_optee_os', 'opcfoundation_ua-.net-legacy', 'opcfoundation_ua-.netstandard', 'open-classifieds_openclassifieds2', 'open-iscsi_tcmu-runner', 'open-power_skiboot', 'open-zaak_open-zaak', 'open5gs_open5gs', 'open62541_open62541', 'openbmc_phosphor-host-ipmid', 'openbsd_src', 'opencart-ce_opencart-ce', 'opencart_opencart', 'opencast_opencast', 'opencats_opencats', 'opencontainers_distribution-spec', 'opencontainers_runc', 'opencontainers_umoci', 'opencrx_opencrx', 'opencv_opencv', 'opendocman_opendocman', 'openemr_openemr', 'openenclave_openenclave', 'opengamepanel_ogp-agent-linux', 'openhab_openhab-addons', 'openidc_pyoidc', 'openid_php-openid', 'openid_ruby-openid', 'openmage_magento-lts', 'openmediavault_openmediavault', 'openmf_mifos-mobile', 'openmpt_openmpt', 'openmrs_openmrs-module-htmlformentry', 'openmrs_openmrs-module-reporting', 'opennms_opennms', 'openolat_openolat', 'openpgpjs_openpgpjs', 'openrc_openrc', 'openresty_lua-nginx-module', 'opensc_opensc', 'opensc_pam_p11', 'opensearch-project_opensearch-cli', 'openshift_console', 'openshift_origin-server', 'opensips_opensips', 'openslides_openslides', 'openssh_openssh-portable', 'openssl_openssl', 'openstack-infra_puppet-gerrit', 'openstack_glance', 'openstack_heat-templates', 'openstack_horizon', 'openstack_keystone', 'openstack_nova-lxd', 'openstack_nova', 'openstack_swauth', 'openstack_swift', 'opensuse_kiwi', 'opensuse_libsolv', 'opensuse_obs-service-set_version', 'opensuse_open-build-service', 'openthread_openthread', 'openthread_wpantund', 'openvpn_openvpn', 'openvswitch_ovs', 'openwhyd_openwhyd', 'openwrt_luci', 'openwrt_openwrt', 'openzeppelin_openzeppelin-contracts', 'openzfs_zfs', 'opf_openproject', 'opmantek_open-audit', 'opnsense_core', 'opscode_chef', 'orbeon_orbeon-forms', 'orchardcms_orchardcore', 'orchidsoftware_platform', 'orckestra_c1-cms-foundation', 'orientechnologies_orientdb', 'oroinc_platform', 'oryx-embedded_cyclonetcp', 'ory_fosite', 'ory_hydra', 'ory_oathkeeper', 'os4ed_opensis-classic', 'os4ed_opensis-responsive-design', 'osclass_osclass', 'osgeo_gdal', 'osquery_osquery', 'ossec_ossec-wui', 'osticket_osticket-1.8', 'osticket_osticket', 'otrs_faq', 'otrs_otrs', 'owasp_json-sanitizer', 'owen2345_camaleon-cms', 'owncloud_apps', 'owncloud_core', 'owncloud_gallery', 'owntone_owntone-server', 'oyejorge_gpeasy-cms', 'oyvindkinsey_easyxdm', 'pac4j_pac4j', 'pagekit_pagekit', 'pahen_madge', 'pallets_jinja', 'pallets_werkzeug', 'paramiko_paramiko', 'paritytech_frontier', 'paritytech_libsecp256k1', 'paritytech_parity', 'parse-community_parse-server', 'patriksimek_vm2', 'patrowl_patrowlmanager', 'paulusmack_ppp', 'pbatard_rufus', 'pcmacdon_jsish', 'pderksen_wp-google-calendar-events', 'pear_archive_tar', 'pediapress_pyfribidi', 'peerigon_angular-expressions', 'peopledoc_vault-cli', 'perl5-dbi_dbd-mysql', 'perl5-dbi_dbi', 'perl_perl5', 'perwendel_spark', 'peterbraden_node-opencv', 'petl-developers_petl', 'pfsense_freebsd-ports', 'pfsense_pfsense', 'pgbouncer_pgbouncer', 'pgjdbc_pgjdbc', 'pgpartman_pg_partman', 'phaag_nfdump', 'phenx_php-font-lib', 'philippk-de_collabtive', 'phoronix-test-suite_phoronix-test-suite', 'phpbb_phpbb', 'phpbb_phpbb3', 'phpfusion_phpfusion', 'phpipam_phpipam', 'phpmailer_phpmailer', 'phpmussel_phpmussel', 'phpmyadmin_phpmyadmin', 'phpoffice_phpspreadsheet', 'phppgadmin_phppgadmin', 'phpservermon_phpservermon', 'phpsocialnetwork_phpfastcache', 'php_php-src', 'phusion_passenger', 'pi-hole_adminlte', 'pi-hole_pi-hole', 'pichi-router_pichi', 'pierrerambaud_gemirro', 'pikepdf_pikepdf', 'pillarjs_resolve-path', 'pillys_fs-path', 'pimcore_pimcore', 'pingidentity_mod_auth_openidc', 'pion_dtls', 'piranhacms_piranha.core', 'piranna_linux-cmdline', 'pires_go-proxyproto', 'piwigo_localfileseditor', 'piwigo_piwigo', 'pixelb_coreutils', 'pixeline_bugs', 'pjshumphreys_patchmerge', 'pjsip_pjproject', 'pkp_omp', 'pksunkara_inflect', 'plataformatec_simple_form', 'pligg_pligg-cms', 'plone_products.cmfplone', 'plone_products.isurlinportal', 'plougher_squashfs-tools', 'pluck-cms_pluck', 'pluginsglpi_addressing', 'pluginsglpi_barcode', 'podlove_podlove-publisher', 'poezio_slixmpp', 'pofider_phantom-html-to-pdf', 'pojome_activity-log', 'polarssl_polarssl', 'pornel_pngquant', 'portainer_portainer', 'postcss_postcss', 'postgres_postgres', 'pow-auth_pow_assent', 'powerdns_pdns', 'prasathmani_tinyfilemanager', 'prestashop_contactform', 'prestashop_dashproducts', 'prestashop_prestashop', 'prestashop_productcomments', 'prestashop_ps_emailsubscription', 'prestashop_ps_facetedsearch', 'prestashop_ps_linklist', 'prestashop_ps_socialfollow', 'prisma-labs_graphql-playground', 'prismjs_prism', 'pritunl_pritunl-client-electron', 'privacyidea_privacyidea', 'privatebin_privatebin', 'processone_ejabberd', 'proftpd_proftpd', 'progval_limnoria', 'project-pier_projectpier-core', 'projectacrn_acrn-hypervisor', 'projectatomic_bubblewrap', 'projectatomic_libpod', 'projectcontour_contour', 'projectsend_projectsend', 'projen_projen', 'prometheus_prometheus', 'protonmail_webclient', 'psi-im_iris', 'pslegr_core-1', 'psychobunny_nodebb-plugin-blog-comments', 'pterodactyl_panel', 'pterodactyl_wings', 'pts_sam2p',
    'publify_publify', 'pugjs_pug', 'pulp_pulp', 'puma_puma', 'punbb_punbb', 'puncsky_touchbase.ai', 'punkave_sanitize-html', 'pupnp_pupnp', 'puppetlabs_mcollective-sshkey-security', 'puppetlabs_puppet', 'puppetlabs_puppetlabs-cinder', 'pusher_oauth2_proxy', 'pwndoc_pwndoc', 'pyca_cryptography', 'pydio_pydio-core', 'pyeve_eve', 'pygments_pygments', 'pylons_waitress', 'pypa_pipenv', 'pyradius_pyrad', 'pytest-dev_py', 'python-discord_bot', 'python-imaging_pillow', 'python-pillow_pillow', 'python-restx_flask-restx', 'python_cpython', 'python_typed_ast', 'pytorchlightning_pytorch-lightning', 'pytroll_donfig', 'q2a-projects_q2a-ultimate-seo', 'q2a_question2answer', 'qbittorrent_qbittorrent', 'qemu_qemu', 'qix-_color-string', 'qos-ch_slf4j', 'qpdf_qpdf', 'qt_qtbase', 'qt_qtsvg', 'quadule_colorscore', 'quagga_quagga', 'quassel_quassel', 'qutebrowser_qutebrowser', 'rackerlabs_openstack-guest-agents-windows-xenserver', 'racket_racket', 'racktables_racktables', 'rack_rack', 'ractf_core', 'radarcovid_radar-covid-android', 'radarcovid_radar-covid-backend-dp3t-server', 'radarcovid_radar-covid-ios', 'radareorg_radare2-extras', 'radareorg_radare2', 'radare_radare2', 'railsdog_spree', 'rails_rails-html-sanitizer', 'rails_rails', 'rails_sprockets', 'rainlab_blog-plugin', 'rainlab_debugbar-plugin', 'rainlab_user-plugin', 'rainloop_rainloop-webmail', 'rakibtg_docker-web-gui', 'randombit_botan', 'rap2hpoutre_laravel-log-viewer', 'rapid7_metasploit-framework', 'rasahq_rasa', 'ratpack_ratpack', 'ravibpatel_autoupdater.net', 'rawstudio_rawstudio', 'rconfig_rconfig', 'rcook_rgpg', 'rdesktop_rdesktop', 'rdoc_rdoc', 'readytalk_avian', 'realtimeprojects_quixplorer', 'recurly_recurly-client-net', 'recurly_recurly-client-python', 'recurly_recurly-client-ruby', 'reddit_snudown', 'redis-store_redis-store', 'redis_hiredis', 'redis_redis', 'redmine_redmine', 'redon-tech_roblox-purchasing-hub', 'reference-lapack_lapack', 'reg-viz_reg-suit', 'rejetto_hfs2', 'relan_exfat', 'relekang_django-nopassword', 'relic-toolkit_relic', 'remarkjs_remark-html', 'remy_undefsafe', 'renlok_webid', 'replit_crosis', 'requarks_wiki', 'requests_requests', 'resiprocate_resiprocate', 'retke_laggrons-dumb-cogs', 'reubenhwk_radvd', 'reviewboard_reviewboard', 'revive-adserver_revive-adserver', 'rgrove_sanitize', 'rhalff_dot-object', 'rhuss_jolokia', 'rhysd_shiba', 'riot-os_riot', 'rizinorg_rizin', 'rjbs_email-address', 'rjmackay_ushahidi_web', 'rkesters_gnuplot', 'robertbachmann_openbsd-libssl', 'robiso_wondercms', 'robo-code_robocode', 'robrichards_xmlseclibs', 'rocketchat_rocket.chat', 'rodnaph_sockso', 'roehling_postsrsd', 'roest01_node-pdf-image', 'rofl0r_proxychains-ng', 'rohe_pysaml2', 'ronf_asyncssh', 'ronomon_opened', 'ronsigal_resteasy', 'root-project_root', 'roundcube_roundcubemail', 'rpm-software-management_libcomps', 'rpm-software-management_rpm', 'rpm-software-management_yum-utils', 'rra_pam-krb5', 'rsantamaria_papercrop', 'rsyslog_rsyslog', 'rs_node-netmask', 'rtomayko_rack-cache', 'ruby-grape_grape', 'ruby-ldap_ruby-net-ldap', 'rubygems_rubygems', 'ruby_openssl', 'ruby_ruby', 'ruby_webrick', 'rumkin_keyget', 'rundeck_rundeck', 'ruscur_linux', 'russellhaering_goxmldsig', 'rust-blockchain_evm', 'rust-lang_mdbook', 'rust-lang_rust', 'rweather_noise-java', 'rxtur_blogengine.net', 's-cart_core', 's-cart_s-cart', 's-gv_orangeforum', 's3131212_allendisk', 's9y_serendipity', 'sabberworm_php-css-parser', 'sabnzbd_sabnzbd', 'sagemath_sagecell', 'saitoha_libsixel', 'salesagility_suitecrm', 'salesforce_tough-cookie', 'salopensource_sal', 'saltstack_salt', 'sama34_ougc-feedback', 'samatt_herbivore', 'samholmes_node-open-graph', 'samtools_htslib', 'samuelcolvin_pydantic', 'sandervanvugt_systemd', 'sandhje_vscode-phpmd', 'sandstorm-io_sandstorm', 'sapplica_sentrifugo', 'sap_less-openui5', 'saschahauer_barebox', 'sassoftware_go-rpmutils', 'sass_libsass', 'sauruscms_saurus-cms-community-edition', 'scandipwa_create-magento-app', 'scaron_prettyphoto', 'schedmd_slurm', 'schine_mw-oauth2client', 'schneems_wicked', 'schokokeksorg_freewvs', 'scipy_scipy', 'scrapy-plugins_scrapy-splash', 'scrapy_scrapy', 'scratchaddons_scratchaddons', 'scratchverifier_scratchoauth2', 'scratchverifier_scratchverifier', 'scravy_node-macaddress', 'sctplab_usrsctp', 'sddm_sddm', 'seam2_jboss-seam', 'sebastianbergmann_phpunit', 'sebhildebrandt_systeminformation', 'sebsauvage_shaarli', 'seccomp_libseccomp-golang', 'secureauthcorp_impacket', 'security-onion-solutions_securityonion', 'sefrengo-cms_sefrengo-1.x', 'selinuxproject_selinux', 'semantic-release_semantic-release', 'semplon_genixcms', 'senchalabs_connect', 'sequelize_sequelize', 'serenityos_serenity', 'serghey-rodin_vesta', 'serpicoproject_serpico', 'servicestack_servicestack', 'sferik_rails_admin', 'sgminer-dev_sgminer', 'shadow-maint_shadow', 'shadowsocks_shadowsocks-libev', 'sharetribe_sharetribe', 'sharkdp_bat', 'sharpred_deephas', 'shellinabox_shellinabox', 'shelljs_shelljs', 'shirasagi_shirasagi', 'shopizer-ecommerce_shopizer', 'shopware_platform', 'shopware_shopware', 'shrinerb_shrine', 'shuox_acrn-hypervisor', 'shuup_shuup', 'siacs_conversations', 'sickrage_sickrage', 'sidhpurwala-huzaifa_freerdp', 'sigil-ebook_sigil', 'signalapp_signal-desktop', 'signalapp_signal-ios', 'signalwire_freeswitch', 'silnrsi_graphite', 'silverstripe_sapphire', 'silverstripe_silverstripe-cms', 'silverstripe_silverstripe-framework', 'silverstripe_silverstripe-installer', 'simonhaenisch_md-to-pdf', 'simpleledger_electron-cash-slp', 'simpleledger_slp-validate', 'simpleledger_slp-validate.js', 'simpleledger_slpjs', 'simplemachines_smf2.1', 'simplesamlphp_saml2', 'simplesamlphp_simplesamlphp', 'simsong_tcpflow', 'sinatra_sinatra', 'sindresorhus_semver-regex', 'sipa_bitcoin', 'sipcapture_homer-app', 'sismics_docs', 'sitaramc_gitolite', 'skalenetwork_sgxwallet', 'skoranga_node-dns-sync', 'skvadrik_re2c', 'skylot_jadx', 'skytable_skytable', 'slackero_phpwcms', 'sleuthkit_sleuthkit', 'slicer69_doas', 'smartstore_smartstorenet', 'smarty-php_smarty', 'snapcore_snapd', 'snapcore_snapweb', 'snipe_snipe-it', 'snorby_snorby', 'socketio_engine.io-client', 'socketio_engine.io', 'socketio_socket.io-parser', 'socketio_socket.io', 'sockjs_sockjs-node', 'softwaremill_akka-http-session', 'soketi_soketi', 'solidusio_solidus', 'solidusio_solidus_auth_devise', 'solj_bcfg2', 'solokeys_solo', 'sonarsource_sonarqube', 'sonicdoe_ced', 'sonicdoe_detect-character-encoding', 'sonnyp_json8', 'sorcery_sorcery', 'soruly_whatanime.ga', 'sosreport_sos-collector', 'sosreport_sos', 'sourcefabric_newscoop', 'sourcegraph_sourcegraph', 'spacewalkproject_spacewalk', 'sparc_phpwhois.org', 'sparkdevnetwork_rock', 'sparklemotion_mechanize', 'sparklemotion_nokogiri', 'spdk_spdk', 'spiderlabs_modsecurity', 'spip_spip', 'spiral-project_ihatemoney', 'splitbrain_dokuwiki', 'sporkmonger_addressable', 'spree_spree', 'spree_spree_auth_devise', 'spring-projects_spring-data-jpa', 'spring-projects_spring-framework', 'sqlcipher_sqlcipher', 'sqlite_sqlite', 'squaresquash_web', 'square_go-jose', 'square_okhttp', 'square_retrofit', 'squid-cache_squid', 'sroehrl_neoan3-template', 'sroracle_abuild', 'ssbc_ssb-db', 'sshock_afflibv3', 'ssnau_killport', 'stachenov_quazip', 'stanfordnlp_corenlp', 'star7th_showdoc', 'status-im_react-native-desktop', 'stdonato_glpi-dashboard', 'stedolan_jq', 'stefanesser_suhosin', 'stephane_libmodbus', 'stevegraham_slanger', 'stevenweathers_thunderdome-planning-poker', 'stnoonan_spnego-http-auth-nginx-module', 'stoth68000_media-tree', 'strangerstudios_paid-memberships-pro', 'strikeentco_set', 'strukturag_libheif', 'studio-42_elfinder', 'stuk_jszip', 'substack_node-shell-quote', 'substack_node-syntax-error', 'sullo_nikto', 'sulu_sulu', 'sup-heliotrope_sup', 'supervisor_supervisor', 'supportflow_supportflow', 'surveysolutions_surveysolutions', 'sustainsys_saml2', 'sveltejs_language-tools', 'svenfuchs_i18n', 'svenfuchs_safemode', 'swagger-api_swagger-codegen', 'swisspol_gcdwebserver', 'swoole_swoole-src', 'sylius_paypalplugin', 'sylius_sylius', 'symfony_debug', 'symfony_security-http', 'symfony_symfony', 'symless_synergy-core', 'symphonycms_symphony-2', 'syncthing_syncthing', 'syoyo_tinyexr', 'sysphonic_thetis', 'systemd_systemd-stable', 'systemd_systemd', 'szukw000_openjpeg', 'tabatkins_bikeshed', 'tadashi-aikawa_owlmixin', 'taglib_taglib', 'tats_w3m', 'tbeu_matio', 'tcltk_tcl', 'tecnickcom_tcpdf', 'teeaykay_moxieplayer', 'teejee2008_timeshift', 'teeworlds_teeworlds', 'tekmonksgithub_monkshu', 'telegramdesktop_tdesktop', 'tematres_tematres-vocabulary-server', 'tenable_integration-jira-cloud', 'tenancy_multi-tenant', 'tendermint_tendermint', 'tensorflow_tensorflow', 'tesseract-ocr_tesseract', 'testlinkopensourcetrms_testlink-code', 'tex-live_texlive-source', 'textpattern_textpattern', 'thanethomson_mlalchemy', 'the-tcpdump-group_libpcap', 'the-tcpdump-group_tcpdump', 'theforeman_foreman', 'theforeman_smart-proxy', 'thehive-project_cortex', 'themoken_canto-curses', 'theonedev_onedev', 'thephpleague_flysystem', 'theupdateframework_python-tuf', 'theupdateframework_tuf', 'thi-ng_umbrella', 'thingsboard_thingsboard', 'thlorenz_parse-link-header', 'thomasdickey_lynx-snapshots', 'thomasdickey_xterm-snapshots', 'thorsten_phpmyfaq', 'thoughtbot_administrate', 'thoughtbot_paperclip', 'tiangolo_fastapi', 'tidwall_gjson', 'tigervnc_tigervnc', 'tikiorg_tiki', 'tillkamppeter_ippusbxd', 'tine20_tine-2.0-open-source-groupware-and-crm', 'tinymce_tinymce_spellchecker_php', 'tj_node-querystring', 'tlhunter_neoinvoice', 'tlsfuzzer_tlslite-ng', 'tmate-io_tmate-ssh-server', 'tmercswims_tmerc-cogs', 'tmux_tmux', 'toastdriven_django-tastypie', 'togatech_tenvoy', 'tojocky_node-printer', 'tomhughes_libdwarf', 'tomoh1r_ansible-vault', 'tony-tsx_cookiex-deep', 'tootallnate_node-degenerator', 'top-think_thinkphp', 'tornadoweb_tornado', 'torproject_tor', 'tortoise_tortoise-orm', 'torvalds_linux', 'totaljs_cms', 'totaljs_framework', 'totaljs_framework4', 'traccar_traccar', 'traefik_traefik',
    'trannamtrung1st_elfinder.net.core', 'transloadit_uppy', 'transmission_transmission', 'tremor-rs_tremor-runtime', 'trestleadmin_trestle-auth', 'trevp_tlslite', 'trgil_gilcc', 'triaxtec_openapi-python-client', 'tribe29_checkmk', 'tridentli_pitchfork', 'troglobit_ssdp-responder', 'troglobit_uftpd', 'trollepierre_tdm', 'tryghost_express-hbs', 'ttimot24_horizontcms', 'turbovnc_turbovnc', 'turistforeningen_node-im-metadata', 'turistforeningen_node-im-resize', 'turquoiseowl_i18n', 'twangboy_salt', 'twigphp_twig', 'twisted_twisted', 'twitter_secure_headers', 'twitter_twitter-server', 'tyktechnologies_tyk-identity-broker', 'typedproject_tsed', 'typesetter_typesetter', 'typestack_class-transformer', 'typo3_fluid', 'typo3_typo3.cms', 'typo3_typo3', 'u-boot_u-boot', 'ua-parser_uap-core', 'uberfire_uberfire', 'ucbrise_opaque', 'uclouvain_openjpeg', 'ueno_libfep', 'ulikunitz_xz', 'ulterius_server', 'ultimatemember_ultimatemember', 'umbraco_umbraco-cms', 'unaio_una', 'unbit_uwsgi', 'unetworking_uwebsockets', 'unicode-org_icu', 'unicorn-engine_unicorn', 'uninett_mod_auth_mellon', 'univention_univention-corporate-server', 'universal-omega_dynamicpagelist3', 'unrealircd_unrealircd', 'unrud_radicale', 'unshiftio_url-parse', 'upx_upx', 'uriparser_uriparser', 'urllib3_urllib3', 'userfrosting_userfrosting', 'ushahidi_ushahidi_web', 'uwebsockets_uwebsockets', 'uyuni-project_uyuni', 'uzbl_uzbl', 'vadz_libtiff', 'validatorjs_validator.js', 'valvesoftware_gamenetworkingsockets', 'vanillaforums_garden', 'vanilla_vanilla', 'vapor_vapor', 'varnishcache_varnish-cache', 'varnish_varnish-cache', 'vega_vega', 'veracrypt_veracrypt', 'vercel_next.js', 'verdammelt_tnef', 'verot_class.upload.php', 'vesse_node-ldapauth-fork', 'veyon_veyon', 'viabtc_viabtc_exchange_server', 'victoralagwu_cmssite', 'videojs_video.js', 'videolan_vlc-ios', 'videolan_vlc', 'viewvc_viewvc', 'viking04_merge', 'vim-syntastic_syntastic', 'vim_vim', 'vincentbernat_lldpd', 'vincit_objection.js', 'virustotal_yara', 'visionmedia_send', 'vito_chyrp', 'vmg_redcarpet', 'vmware_xenon', 'volca_markdown-preview', 'voten-co_voten', 'vrana_adminer', 'vrtadmin_clamav-devel', 'vscode-restructuredtext_vscode-restructuredtext', 'vscodevim_vim', 'vslavik_winsparkle', 'vuelidate_vuelidate', 'vulcanjs_vulcan', 'w3c_css-validator', 'w8tcha_ckeditor-oembed-plugin', 'wagtail_wagtail', 'wal-g_wal-g', 'wanasit_chrono', 'wavm_wavm', 'waysact_webpack-subresource-integrity', 'wbce_wbce_cms', 'wbx-github_uclibc-ng', 'weaveworks_weave', 'web2project_web2project', 'web2py_web2py', 'webbukkit_dynmap', 'webkit_webkit', 'weblateorg_weblate', 'webmin_webmin', 'webpack_webpack-dev-server', 'webrecorder_pywb', 'websockets_ws', 'weechat_weechat', 'weidai11_cryptopp', 'wekan_wekan', 'weld_core', 'wernerd_zrtpcpp', 'weseek_growi', 'wesnoth_wesnoth', 'westes_flex', 'wez_atomicparsley', 'wgm_cerb', 'whyrusleeping_tar-utils', 'wichert_pyrad', 'wikimedia_analytics-quarry-web', 'wikimedia_mediawiki-core', 'wikimedia_mediawiki', 'wildfly-security_jboss-negotiation', 'wildfly-security_soteria', 'winscp_winscp', 'wireapp_wire-desktop', 'wireapp_wire-ios-data-model', 'wireapp_wire-ios', 'wireapp_wire-server', 'wireapp_wire-webapp', 'wireshark_wireshark', 'wolfcms_wolfcms', 'wolfssl_wolfmqtt', 'wolfssl_wolfssl', 'wordpress_wordpress-develop', 'wordpress_wordpress', 'wp-plugins_sagepay-direct-for-woocommerce-payment-gateway', 'wp-plugins_w3-total-cache', 'wp-statistics_wp-statistics', 'wpeverest_everest-forms', 'wting_autojump', 'wuyouzhuguli_febs-shiro', 'wwbn_avideo', 'x-stream_xstream', 'x2engine_x2crm', 'xbmc_xbmc', 'xcllnt_openiked', 'xcritical-software_utilitify', 'xelerance_openswan', 'xenocrat_chyrp-lite', 'xfce-mirror_thunar', 'xianyi_openblas', 'xjodoin_torpedoquery', 'xkbcommon_libxkbcommon', 'xmidt-org_cjwt', 'xmldom_xmldom', 'xoops_xoopscore25', 'xrootd_xrootd', 'xsmo_image-uploader-and-browser-for-ckeditor', 'xwiki-labs_cryptpad', 'xwiki_xwiki-platform', 'yahoo_elide', 'yahoo_serialize-javascript', 'yanvugenfirer_kvm-guest-drivers-windows', 'yarnpkg_website', 'yarnpkg_yarn', 'yarolig_didiwiki', 'yast_yast-core', 'yeraze_ytnef', 'yetiforcecompany_yetiforcecrm', 'yhatt_jsx-slack', 'yhirose_cpp-peglib', 'yiisoft_yii2', 'yoast_wordpress-seo', 'yoe_nbd', 'yogeshojha_rengine', 'youphptube_youphptube', 'yourls_yourls', 'ysurac_openmptcprouter-vps-admin', 'yubico_libu2f-host', 'yubico_pam-u2f', 'yubico_yubico-pam', 'z3apa3a_3proxy', 'zammad_zammad', 'zblogcn_zblogphp', 'zcash_zcash', 'zencart-ja_zc-v1-series', 'zendframework_zf2', 'zenphoto_zenphoto', 'zephyrkul_fluffycogs', 'zephyrproject-rtos_zephyr', 'zeroclipboard_zeroclipboard', 'zeromq_libzmq', 'zeromq_zeromq4-x', 'zeroturnaround_zt-zip', 'zf-commons_zfcuser', 'zfsonlinux_zfs', 'zhaozg_lua-openssl', 'zherczeg_jerryscript', 'zhutougg_c3p0', 'zimbra_zm-mailbox', 'zjonsson_node-unzipper', 'zkat_ssri', 'zmartzone_mod_auth_openidc', 'zmister2016_mrdoc', 'znc_znc', 'zom_zom-ios', 'zoneminder_zoneminder', 'zopefoundation_accesscontrol', 'zopefoundation_products.cmfcore', 'zopefoundation_products.genericsetup', 'zopefoundation_products.pluggableauthservice', 'zopefoundation_zope', 'ztree_ztree_v3', 'zulip_zulip', 'zyantific_zydis']
