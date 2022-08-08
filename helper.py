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
            raise ValueError("type must be assigned an Enum when using EnumAction")
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
        precision, recall, fscore, support = f_score(y_test, y_predict, zero_division=0)
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
    reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) + np.std(reconstruction_errors.numpy())
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

def run_query(query,ignore_errors=False):
    counter = 0;
    while True:
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
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

            err_string = "Query failed to run by returning code of {}. {}".format(request.status_code, query)
            if ignore_errors:
                print(err_string)
            else:
                raise Exception(err_string)



def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

from functools import wraps
from time import time

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