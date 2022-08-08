#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.callbacks import EarlyStopping
from models import conv1d, conv1d_more_layers, lstm, small_conv1d
from sklearn.metrics import roc_curve, auc
from helper import find_best_f1, EnumAction, safe_mkdir
from classes import Repository
from matplotlib import pyplot as plt
from matplotlib import pyplot
from enum import Enum
from pandas import DataFrame
import helper
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import matplotlib
import datetime
import random
import tqdm
import pickle
import json
import os
import logging

import coloredlogs
import logging
logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')


matplotlib.use('Agg')


BENIGN_TAG = 0
VULN_TAG = 1


def find_benign_events(cur_repo_data, gap_days, num_of_events):
    """
    :param cur_repo_data: DataFrame that is processed
    :param gap_days: number of days to look back for the events
    :param num_of_events: number of events to find
    :return: list of all events
    """
    benign_events = []
    retries = num_of_events * 5
    counter = 0
    for _ in range(num_of_events):
        found_event = False
        while not found_event:
            if counter >= retries:
                return benign_events
            try:
                cur_event = random.randint(
                    2 * gap_days + 1, cur_repo_data.shape[0] - gap_days * 2 - 1)
            except ValueError:
                counter += 1
                continue
            event = cur_repo_data.index[cur_event]

            before_vuln = event - gap_days
            after_vuln = event + gap_days
            res_event = cur_repo_data[before_vuln:event - 1]
            if not res_event[res_event["VulnEvent"] > 0].empty:
                counter += 1
                continue
            benign_events.append(res_event.iloc[:, :-1].values)
            found_event = True

    return benign_events


def create_all_events(cur_repo_data, gap_days):
    """
    :param cur_repo_data: DataFrame that is processed
    :param gap_days: number of days to look back for the events
    :return: list of all events
    """
    all_events = []
    labels = []
    for i in range(gap_days, cur_repo_data.shape[0], 1):
        event = cur_repo_data.index[i]
        before_vuln = event - gap_days
        res_event = cur_repo_data[before_vuln:event - 1]
        all_events.append(res_event.iloc[:, :-1].values)
        labels.append(res_event.iloc[:, -1].values)
    return all_events, labels


event_types = ['PullRequestEvent', 'PushEvent', 'ReleaseEvent', 'DeleteEvent', 'issues', 'CreateEvent', 'releases', 'IssuesEvent', 'ForkEvent', 'WatchEvent', 'PullRequestReviewCommentEvent',
               'stargazers', 'pullRequests', 'commits', 'CommitCommentEvent', 'MemberEvent', 'GollumEvent', 'IssueCommentEvent', 'forks', 'PullRequestReviewEvent', 'PublicEvent']


def add_type_one_hot_encoding(df):
    """
    :param df: dataframe to add type one hot encoding to
    :return: dataframe with type one hot encoding
    """
    type_one_hot = pd.get_dummies(df.type.astype(
        pd.CategoricalDtype(categories=event_types)))
    df = pd.concat([df, type_one_hot], axis=1)
    return df


def add_time_one_hot_encoding(df, with_idx=False):
    """
    :param df: dataframe to add time one hot encoding to
    :param with_idx: if true, adds index column to the dataframe
    :return: dataframe with time one hot encoding
    """

    hour = pd.get_dummies(df.index.get_level_values(0).hour.astype(pd.CategoricalDtype(categories=range(24))),
                          prefix='hour')
    week = pd.get_dummies(df.index.get_level_values(0).dayofweek.astype(pd.CategoricalDtype(categories=range(7))),
                          prefix='day_of_week')
    day_of_month = pd.get_dummies(df.index.get_level_values(0).day.astype(pd.CategoricalDtype(categories=range(1, 32))),
                                  prefix='day_of_month')

    df = pd.concat([df.reset_index(), hour, week, day_of_month], axis=1)
    if with_idx:
        df = df.set_index(['created_at', 'idx'])
    else:
        df = df.set_index(['index'])
    return df


def get_event_window(cur_repo_data, event, aggr_options, days=10, hours=10, backs=50, resample=24):
    """
    :param cur_repo_data: DataFrame that is processed
    :param event: list of events to get windows from
    :param aggr_options: can be before, after or none, to decide how we agregate
    :param days: if 'before' or 'after' is choosed as aggr_options
        amount of days gathered as a single window (in addition to hours)
    :param hours: if 'before' or 'after' is choosed as aggr_options
        amount of hours gathered as a single window (in addition to days)
    :param backs: if 'none' is choosed as aggr_options, this is the amount of events back taken
    :param resample: is the data resampled and at what frequency (hours)
    :return: a window for lstm
    """
    befs = -1
    hours_befs = 2

    if aggr_options == Aggregate.after_cve:
        cur_repo_data = cur_repo_data.reset_index().drop(
            ["idx"], axis=1).set_index("created_at")
        cur_repo_data = cur_repo_data.sort_index()
        indicator = event[0] - datetime.timedelta(days=0, hours=hours_befs)
        starting_time = indicator - datetime.timedelta(days=days, hours=hours)
        res = cur_repo_data[starting_time:indicator]
        new_row = pd.DataFrame([[0] * len(res.columns)],
                               columns=res.columns, index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        res = res.resample(f'{resample}H').sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.before_cve:
        res = cur_repo_data.reset_index().drop(["created_at"], axis=1).set_index("idx")[
        event[1] - backs:event[1] + backs]   

    elif aggr_options == Aggregate.none:
        res = cur_repo_data.reset_index().drop(["created_at"], axis=1).set_index("idx")[
            event[1] - backs:event[1] + befs]
    return res.values


repo_dirs = 'hiddenCVE/gh_cve_proccessed'
benign_all, vuln_all = [], []
n_features = 0
gap_days = 150

nice_list = ['facebook_hhvm.csv',
             'ffmpeg_ffmpeg.csv',
             'flatpak_flatpak.csv',
             'freerdp_freerdp.csv',
             'git_git.csv',
             'gpac_gpac.csv',
             'imagemagick_imagemagick.csv',
             'kde_kdeconnect-kde.csv',
             'krb5_krb5.csv',
             'mantisbt_mantisbt.csv',
             'op-tee_optee_os.csv',
             # 'owncloud_core.csv',
             'php_php-src.csv',
             'revive-adserver_revive-adserver.csv',
             # 'rubygems_rubygems.csv',
             # 'the-tcpdump-group_tcpdump.csv'
             ]

nice_list2 = ['abrt_abrt.csv', 'clusterlabs_pcs.csv', 'discourse_discourse.csv', 'exponentcms_exponent-cms.csv',
              'facebook_hhvm.csv', 'ffmpeg_ffmpeg.csv', 'file_file.csv', 'firefly-iii_firefly-iii.csv',
              'flatpak_flatpak.csv', 'freerdp_freerdp.csv', 'fusionpbx_fusionpbx.csv', 'git_git.csv', 'gpac_gpac.csv',
              'ifmeorg_ifme.csv', 'imagemagick_imagemagick.csv', 'jenkinsci_jenkins.csv', 'kanboard_kanboard.csv',
              'kde_kdeconnect-kde.csv', 'koral--_android-gif-drawable.csv', 'krb5_krb5.csv',
              'libarchive_libarchive.csv', 'libgit2_libgit2.csv', 'libraw_libraw.csv',
              'livehelperchat_livehelperchat.csv', 'mantisbt_mantisbt.csv', 'mdadams_jasper.csv', 'oisf_suricata.csv',
              'op-tee_optee_os.csv', 'openssh_openssh-portable.csv', 'openssl_openssl.csv', 'owncloud_core.csv',
              'php_php-src.csv']

less_than_10_vulns = ['01org_opa-ff.csv', '01org_opa-fm.csv', '01org_tpm2.0-tools.csv', '10gen-archive_mongo-c-driver-legacy.csv', '1up-lab_oneupuploaderbundle.csv', '389ds_389-ds-base.csv', '3s3s_opentrade.csv', '94fzb_zrlog.csv', 'aaron-junker_usoc.csv', 'aaugustin_websockets.csv', 'aawc_unrar.csv', 'abcprintf_upload-image-with-ajax.csv', 'abhinavsingh_proxy.py.csv', 'absolunet_kafe.csv', 'acassen_keepalived.csv', 'accel-ppp_accel-ppp.csv', 'accenture_mercury.csv', 'acinq_eclair.csv', 'acossette_pillow.csv', 'acpica_acpica.csv', 'actions_http-client.csv', 'adaltas_node-csv-parse.csv', 'adaltas_node-mixme.csv', 'adamghill_django-unicorn.csv', 'adamhathcock_sharpcompress.csv', 'adaptivecomputing_torque.csv', 'admidio_admidio.csv', 'adodb_adodb.csv', 'adrienverge_openfortivpn.csv', 'advancedforms_advanced-forms.csv', 'afarkas_lazysizes.csv', 'ahdinosaur_set-in.csv', 'aheckmann_mpath.csv', 'aheckmann_mquery.csv', 'aimhubio_aim.csv', 'aio-libs_aiohttp.csv', 'aircrack-ng_aircrack-ng.csv', 'airmail_airmailplugin-framework.csv', 'airsonic_airsonic.csv', 'ai_nanoid.csv', 'akashrajpurohit_clipper.csv', 'akheron_jansson.csv', 'akimd_bison.csv', 'akrennmair_newsbeuter.csv', 'alanaktion_phproject.csv', 'alandekok_freeradius-server.csv', 'alanxz_rabbitmq-c.csv', 'albertobeta_podcastgenerator.csv', 'alerta_alerta.csv', 'alexreisner_geocoder.csv', 'alex_rply.csv', 'algolia_algoliasearch-helper-js.csv', 'alkacon_apollo-template.csv', 'alkacon_mercury-template.csv', 'alkacon_opencms-core.csv', 'amazeeio_lagoon.csv', 'ambiot_amb1_arduino.csv', 'ambiot_amb1_sdk.csv', 'ampache_ampache.csv', 'amyers634_muracms.csv', 'anchore_anchore-engine.csv', 'andialbrecht_sqlparse.csv', 'andrerenaud_pdfgen.csv', 'android_platform_bionic.csv', 'andrzuk_finecms.csv', 'andya_cgi--simple.csv', 'andyrixon_layerbb.csv', 'angus-c_just.csv', 'ankane_chartkick.csv', 'ansible-collections_community.crypto.csv', 'ansible_ansible-modules-extras.csv', 'antonkueltz_fastecdsa.csv', 'antswordproject_antsword.csv', 'anurodhp_monal.csv', 'anymail_django-anymail.csv', 'aomediacodec_libavif.csv', 'apache_activemq-artemis.csv', 'apache_activemq.csv', 'apache_cordova-plugin-file-transfer.csv', 'apache_cordova-plugin-inappbrowser.csv', 'apache_cxf-fediz.csv', 'apache_cxf.csv', 'apache_incubator-livy.csv', 'apache_incubator-openwhisk-runtime-docker.csv', 'apache_incubator-openwhisk-runtime-php.csv', 'apache_ofbiz-framework.csv', 'apache_openoffice.csv', 'apache_vcl.csv', 'apexcharts_apexcharts.js.csv', 'apollosproject_apollos-apps.csv', 'apostrophecms_apostrophe.csv', 'apple_cups.csv', 'appneta_tcpreplay.csv', 'aptana_jaxer.csv', 'aquaverde_aquarius-core.csv', 'aquynh_capstone.csv', 'arangodb_arangodb.csv', 'archivy_archivy.csv', 'ardatan_graphql-tools.csv', 'ardour_ardour.csv', 'area17_twill.csv', 'aresch_rencode.csv', 'argoproj_argo-cd.csv', 'arjunmat_slack-chat.csv', 'arrow-kt_arrow.csv', 'arsenal21_all-in-one-wordpress-security.csv', 'arsenal21_simple-download-monitor.csv', 'arslancb_clipbucket.csv', 'artifexsoftware_ghostpdl.csv', 'artifexsoftware_jbig2dec.csv', 'asaianudeep_deep-override.csv', 'ashinn_irregex.csv', 'askbot_askbot-devel.csv', 'assfugil_nickchanbot.csv', 'asteinhauser_fat_free_crm.csv', 'atheme_atheme.csv', 'atheme_charybdis.csv', 'atinux_schema-inspector.csv', 'att_ast.csv', 'auracms_auracms.csv', 'aurelia_path.csv', 'auth0_ad-ldap-connector.csv', 'auth0_auth0.js.csv',
                      'auth0_express-jwt.csv', 'auth0_express-openid-connect.csv', 'auth0_lock.csv', 'auth0_nextjs-auth0.csv', 'auth0_node-auth0.csv', 'auth0_node-jsonwebtoken.csv', 'auth0_omniauth-auth0.csv', 'authelia_authelia.csv', 'authguard_authguard.csv', 'authzed_spicedb.csv', 'automattic_genericons.csv', 'automattic_mongoose.csv', 'autotrace_autotrace.csv', 'autovance_ftp-srv.csv', 'avar_plack.csv', 'avast_retdec.csv', 'awslabs_aws-js-s3-explorer.csv', 'awslabs_tough.csv', 'aws_aws-sdk-js-v3.csv', 'aws_aws-sdk-js.csv', 'axdoomer_doom-vanille.csv', 'axios_axios.csv', 'axkibe_lsyncd.csv', 'b-heilman_bmoor.csv', 'babelouest_glewlwyd.csv', 'babelouest_ulfius.csv', 'bacula-web_bacula-web.csv', 'badongdyc_fangfacms.csv', 'bagder_curl.csv', 'balderdashy_sails-hook-sockets.csv', 'ballerina-platform_ballerina-lang.csv', 'bbangert_beaker.csv', 'bbengfort_confire.csv', 'bblanchon_arduinojson.csv', 'bblfsh_bblfshd.csv', 'bcfg2_bcfg2.csv', 'bcit-ci_codeigniter.csv', 'bcosca_fatfree-core.csv', 'bdew-minecraft_bdlib.csv', 'beanshell_beanshell.csv', 'behdad_harfbuzz.csv', 'belledonnecommunications_belle-sip.csv', 'belledonnecommunications_bzrtp.csv', 'benjaminkott_bootstrap_package.csv', 'bertramdev_asset-pipeline.csv', 'bettererrors_better_errors.csv', 'billz_raspap-webgui.csv', 'bit-team_backintime.csv', 'bitcoin_bitcoin.csv', 'bitlbee_bitlbee.csv', 'bitmessage_pybitmessage.csv', 'bittorrent_bootstrap-dht.csv', 'blackcatdevelopment_blackcatcms.csv', 'blackducksoftware_hub-rest-api-python.csv', 'blogifierdotnet_blogifier.csv', 'blogotext_blogotext.csv', 'blosc_c-blosc2.csv', 'bludit_bludit.csv', 'blueness_sthttpd.csv', 'bluez_bluez.csv', 'bminor_bash.csv', 'bminor_glibc.csv', 'bonzini_qemu.csv', 'boonebgorges_buddypress-docs.csv', 'boonstra_slideshow.csv', 'boothj5_profanity.csv', 'bottlepy_bottle.csv', 'bouke_django-two-factor-auth.csv', 'bower_bower.csv', 'boxug_trape.csv', 'bradyvercher_gistpress.csv', 'braekling_wp-matomo.csv', 'bratsche_pango.csv', 'brave_brave-core.csv', 'brave_muon.csv', 'briancappello_flask-unchained.csv', 'brocaar_chirpstack-network-server.csv', 'broofa_node-uuid.csv', 'brookinsconsulting_bccie.csv', 'browserless_chrome.csv', 'browserslist_browserslist.csv', 'browserup_browserup-proxy.csv', 'bro_bro.csv', 'btcpayserver_btcpayserver.csv', 'buddypress_buddypress.csv', 'bytecodealliance_lucet.csv', 'bytom_bytom.csv', 'c-ares_c-ares.csv', 'c2fo_fast-csv.csv', 'cakephp_cakephp.csv', 'canarymail_mailcore2.csv', 'candlepin_candlepin.csv', 'candlepin_subscription-manager.csv', 'canonicalltd_subiquity.csv', 'caolan_forms.csv', 'capnproto_capnproto.csv', 'carltongibson_django-filter.csv', 'carrierwaveuploader_carrierwave.csv', 'catfan_medoo.csv', 'cauldrondevelopmentllc_cbang.csv', 'ccxvii_mujs.csv', 'cdcgov_microbetrace.csv', 'cdrummond_cantata.csv', 'cdr_code-server.csv', 'ceph_ceph-deploy.csv', 'ceph_ceph-iscsi-cli.csv', 'certtools_intelmq-manager.csv', 'cesanta_mongoose-os.csv', 'cesanta_mongoose.csv', 'cesnet_perun.csv', 'chalk_ansi-regex.csv', 'charleskorn_kaml.csv', 'charybdis-ircd_charybdis.csv', 'chaskiq_chaskiq.csv', 'chatsecure_chatsecure-ios.csv', 'chatwoot_chatwoot.csv', 'check-spelling_check-spelling.csv', 'cherokee_webserver.csv', 'chevereto_chevereto-free.csv', 'chillu_silverstripe-framework.csv', 'chjj_marked.csv', 'chocolatey_boxstarter.csv', 'chopmo_rack-ssl.csv', 'chrisd1100_uncurl.csv', 'chyrp_chyrp.csv',
                      'circl_ail-framework.csv', 'cisco-talos_clamav-devel.csv', 'cisco_thor.csv', 'civetweb_civetweb.csv', 'ckeditor_ckeditor4.csv', 'ckolivas_cgminer.csv', 'claviska_simple-php-captcha.csv', 'clientio_joint.csv', 'cloudendpoints_esp.csv', 'cloudfoundry_php-buildpack.csv', 'clusterlabs_pacemaker.csv', 'cmuir_uncurl.csv', 'cnlh_nps.csv', 'cobbler_cobbler.csv', 'cockpit-project_cockpit.csv', 'codecov_codecov-node.csv', 'codehaus-plexus_plexus-archiver.csv', 'codehaus-plexus_plexus-utils.csv', 'codeigniter4_codeigniter4.csv', 'codemirror_codemirror.csv', 'codiad_codiad.csv', 'cog-creators_red-dashboard.csv', 'cog-creators_red-discordbot.csv', 'collectd_collectd.csv', 'commenthol_serialize-to-js.csv', 'common-workflow-language_cwlviewer.csv', 'composer_composer.csv', 'composer_windows-setup.csv', 'concrete5_concrete5-legacy.csv', 'containers_bubblewrap.csv', 'containers_image.csv', 'containers_libpod.csv', 'containous_traefik.csv', 'contiki-ng_contiki-ng.csv', 'convos-chat_convos.csv', 'cooltey_c.p.sub.csv', 'coreutils_gnulib.csv', 'corosync_corosync.csv', 'cosenary_instagram-php-api.csv', 'cosmos_cosmos-sdk.csv', 'cotonti_cotonti.csv', 'coturn_coturn.csv', 'crater-invoice_crater.csv', 'crawl_crawl.csv', 'creatiwity_witycms.csv', 'creharmony_node-etsy-client.csv', 'crowbar_barclamp-crowbar.csv', 'crowbar_barclamp-deployer.csv', 'crowbar_barclamp-trove.csv', 'crowbar_crowbar-openstack.csv', 'crypto-org-chain_cronos.csv', 'cthackers_adm-zip.csv', 'ctripcorp_apollo.csv', 'ctz_rustls.csv', 'cubecart_v6.csv', 'cure53_dompurify.csv', 'cvandeplas_pystemon.csv', 'cve-search_cve-search.csv', 'cveproject_cvelist.csv', 'cyberark_conjur-oss-helm-chart.csv', 'cyberhobo_wordpress-geo-mashup.csv', 'cydrobolt_polr.csv', 'cyrusimap_cyrus-imapd.csv', 'cyu_rack-cors.csv', 'd0c-s4vage_lookatme.csv', 'd4software_querytree.csv', 'daaku_nodejs-tmpl.csv', 'dagolden_capture-tiny.csv', 'dajobe_raptor.csv', 'daltoniam_starscream.csv', 'dandavison_delta.csv', 'dankogai_p5-encode.csv', 'danschultzer_pow.csv', 'darktable-org_rawspeed.csv', 'darold_squidclamav.csv', 'dart-lang_sdk.csv', 'darylldoyle_svg-sanitizer.csv', 'dashbuilder_dashbuilder.csv', 'datacharmer_dbdeployer.csv', 'datatables_datatablessrc.csv', 'datatables_dist-datatables.csv', 'dav-git_dav-cogs.csv', 'davegamble_cjson.csv', 'davidben_nspluginwrapper.csv', 'davideicardi_confinit.csv', 'davidjclark_phpvms-popupnews.csv', 'daylightstudio_fuel-cms.csv', 'dbeaver_dbeaver.csv', 'dbijaya_onlinevotingsystem.csv', 'dcit_perl-crypt-jwt.csv', 'debiki_talkyard.csv', 'deislabs_oras.csv', 'delta_pragyan.csv', 'delvedor_find-my-way.csv', 'demon1a_discord-recon.csv', 'denkgroot_spina.csv', 'deoxxa_dotty.csv', 'dependabot_dependabot-core.csv', 'derf_feh.csv', 'derickr_timelib.csv', 'derrekr_android_security.csv', 'desrt_systemd-shim.csv', 'deuxhuithuit_symphony-2.csv', 'devsnd_cherrymusic.csv', 'dexidp_dex.csv', 'dgl_cgiirc.csv', 'dhis2_dhis2-core.csv', 'diegohaz_bodymen.csv', 'diegohaz_querymen.csv', 'dieterbe_uzbl.csv', 'digint_btrbk.csv', 'digitalbazaar_forge.csv', 'dingelish_rust-base64.csv', 'dinhviethoa_libetpan.csv', 'dino_dino.csv', 'directus_app.csv', 'directus_directus.csv', 'discourse_discourse-footnote.csv', 'discourse_discourse-reactions.csv', 'discourse_message_bus.csv', 'discourse_rails_multisite.csv', 'diversen_gallery.csv', 'divio_django-cms.csv', 'diygod_rsshub.csv', 'djabberd_djabberd.csv', 'django-helpdesk_django-helpdesk.csv', 'django-wiki_django-wiki.csv', 'dlitz_pycrypto.csv', 'dmendel_bindata.csv', 'dmgerman_ninka.csv', 'dmlc_ps-lite.csv', 'dmproadmap_roadmap.csv', 'dnnsoftware_dnn.platform.csv', 'docker_cli.csv', 'docker_docker-credential-helpers.csv', 'docsifyjs_docsify.csv', 'doctrine_dbal.csv', 'documize_community.csv', 'dogtagpki_pki.csv', 'dojo_dijit.csv', 'dojo_dojo.csv', 'dojo_dojox.csv', 'dollarshaveclub_shave.csv', 'dom4j_dom4j.csv', 'domoticz_domoticz.csv', 'dompdf_dompdf.csv', 'doorgets_doorgets.csv', 'doorkeeper-gem_doorkeeper.csv', 'dosfstools_dosfstools.csv', 'dotcms_core.csv', 'dotse_zonemaster-gui.csv', 'dottgonzo_node-promise-probe.csv', 'dovecot_core.csv', 'doxygen_doxygen.csv', 'dozermapper_dozer.csv', 'dpgaspar_flask-appbuilder.csv', 'dracutdevs_dracut.csv', 'dramforever_vscode-ghc-simple.csv', 'drk1wi_portspoof.csv', 'droolsjbpm_drools.csv', 'droolsjbpm_jbpm-designer.csv', 'droolsjbpm_jbpm.csv', 'droolsjbpm_kie-wb-distributions.csv', 'dropbox_lepton.csv', 'dropwizard_dropwizard.csv', 'drudru_ansi_up.csv', 'dspace_dspace.csv', 'dspinhirne_netaddr-rb.csv', 'dsyman2_integriaims.csv', 'dtschump_cimg.csv', 'duchenerc_artificial-intelligence.csv', 'duffelhq_paginator.csv', 'dukereborn_cmum.csv', 'duncaen_opendoas.csv', 'dutchcoders_transfer.sh.csv', 'dvirtz_libdwarf.csv', 'dweomer_containerd.csv', 'dwisiswant0_apkleaks.csv', 'dw_mitogen.csv', 'dynamoose_dynamoose.csv', 'e107inc_e107.csv', 'e2guardian_e2guardian.csv', 'e2openplugins_e2openplugin-openwebif.csv', 'eclipse-ee4j_mojarra.csv', 'eclipse_mosquitto.csv', 'eclipse_rdf4j.csv', 'eclipse_vert.x.csv', 'edge-js_edge.csv', 'edgexfoundry_app-functions-sdk-go.csv', 'edx_edx-platform.csv', 'eflexsystems_node-samba-client.csv', 'eggjs_extend2.csv', 'egroupware_egroupware.csv', 'eiskalteschatten_compile-sass.csv', 'eivindfjeldstad_dot.csv', 'elabftw_elabftw.csv', 'elastic_elasticsearch.csv', 'eldy_awstats.csv', 'elementary_switchboard-plug-bluetooth.csv', 'elementsproject_lightning.csv', 'elixir-plug_plug.csv', 'ellson_graphviz.csv', 'elmar_ldap-git-backup.csv', 'elric1_knc.csv', 'elves_elvish.csv', 'embedthis_appweb.csv', 'embedthis_goahead.csv', 'emca-it_energy-log-server-6.x.csv', 'emlog_emlog.csv', 'enalean_gitphp.csv', 'enferex_pdfresurrect.csv', 'ensc_irssi-proxy.csv', 'ensdomains_ens.csv', 'enviragallery_envira-gallery-lite.csv', 'envoyproxy_envoy.csv', 'ericcornelissen_git-tag-annotation-action.csv', 'ericcornelissen_shescape.csv', 'ericnorris_striptags.csv', 'ericpaulbishop_gargoyle.csv', 'erikdubbelboer_phpredisadmin.csv', 'erlang_otp.csv', 'erlyaws_yaws.csv', 'esl_mongooseim.csv', 'esnet_iperf.csv', 'esphome_esphome.csv', 'ethereum_go-ethereum.csv', 'ethereum_solidity.csv', 'ether_ueberdb.csv', 'ettercap_ettercap.csv', 'eugeneware_changeset.csv', 'eugeny_ajenti.csv', 'evangelion1204_multi-ini.csv', 'evanphx_json-patch.csv', 'evilnet_nefarious2.csv', 'evilpacket_marked.csv',
                      'excon_excon.csv', 'exiftool_exiftool.csv', 'exim_exim.csv', 'express-handlebars_express-handlebars.csv', 'eyesofnetworkcommunity_eonweb.csv', 'ezsystems_ezjscore.csv', 'f21_jwt.csv', 'fabiocaccamo_utils.js.csv', 'fabpot_twig.csv', 'facebookincubator_fizz.csv', 'facebookincubator_mvfst.csv', 'facebookresearch_parlai.csv', 'facebook_buck.csv', 'facebook_folly.csv', 'facebook_mcrouter.csv', 'facebook_nuclide.csv', 'facebook_react-native.csv', 'facebook_wangle.csv', 'facebook_zstd.csv', 'faisalman_ua-parser-js.csv', 'faiyazalam_wordpress-plugin-user-login-history.csv', 'fardog_trailing-slash.csv', 'fasterxml_jackson-dat']


class Aggregate(Enum):
    none = "none"
    before_cve = "before"
    after_cve = "after"


def create_dataset(aggr_options, benign_vuln_ratio, hours, days, resample, backs):
    """

    :param aggr_options: can be before, after or none, to decide how we agregate
    :param benign_vuln_ratio: ratio of benign to vuln
    :param hours: if 'before' or 'after' is choosed as aggr_options
    :param days:    if 'before' or 'after' is choosed as aggr_options
    :param resample: is the data resampled and at what frequency (hours)
    :param backs: if 'none' is choosed as aggr_options, this is the amount of events back taken
    :return: dataset
    """
    all_repos = []
    all_set = set()
    ignored = []
    dirname = make_new_dir_name(
        aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    safe_mkdir("ready_data")
    safe_mkdir("ready_data/" + dirname)

    counter = 0
    for file in (pbar := tqdm.tqdm(helper.alls[::])):

        counter += 1
        # if counter > 20:
        #     break
        repo_holder = Repository()
        repo_holder.file = file
        pbar.set_description(f"{file} read")

        if not os.path.exists(repo_dirs+"/"+file+".parquet"):
            continue
            try:
                cur_repo = pd.read_csv(
                    repo_dirs + "/" + file+".csv",
                    low_memory=False,
                    parse_dates=['created_at'],
                    dtype={"type": "string", "name": "string", "Hash": "string", "Add":  np.float64, "Del":  np.float64, "Files": np.float64, "Vuln":  np.float64})
            except pd.errors.EmptyDataError:
                continue

            cur_repo.to_parquet(repo_dirs + "/" + file+".parquet")

        else:
            cur_repo = pd.read_parquet(repo_dirs + "/" + file+".parquet")
        
        if cur_repo.shape[0] < 100:
            ignored.append(file)
            continue
        cur_repo["Hash"] = cur_repo["Hash"].fillna("")
        cur_repo = cur_repo.fillna(0)


        number_of_vulns = cur_repo[cur_repo['Vuln'] != 0].shape[0]
        ignored.append((file, number_of_vulns))

        pbar.set_description(f"{file},{number_of_vulns} fix_repo_shape")

        cur_repo = fix_repo_shape(all_set, cur_repo)

        vulns = cur_repo.index[cur_repo['Vuln'] == 1].tolist()
        if not len(vulns):
            continue
        benigns = cur_repo.index[cur_repo['Vuln'] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[:benign_vuln_ratio*len(vulns)]

        cur_repo = cur_repo.drop(["Vuln"],axis=1)
        pbar.set_description(f"{file} extract_window")
        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)

        extract_window(aggr_options, hours, days, resample, backs,
                       file, repo_holder.vuln_lst, repo_holder.vuln_details, cur_repo, vulns, VULN_TAG)

        extract_window(aggr_options, hours, days, resample, backs,
                       file, repo_holder.benign_lst, repo_holder.benign_details, cur_repo, benigns, BENIGN_TAG)

        pbar.set_description(f"{file} pad")

        repo_holder.pad_repo()

        pbar.set_description(f"{file} save")

        with open("ready_data/" + dirname + "/" + repo_holder.file + ".pkl", 'wb') as f:
            pickle.dump(repo_holder, f)

        all_repos.append(repo_holder)

    return all_repos


def extract_window(aggr_options, hours, days, resample, backs, file, window_lst, details_lst, cur_repo, cur_list, tag):
    """
    pulls out a window of events from the repo
    :param aggr_options: can be before, after or none, to decide how we agregate
    :param hours: if 'before' or 'after' is choosed as aggr_options
    :param days:    if 'before' or 'after' is choosed as aggr_options
    :param resample: is the data resampled and at what frequency (hours)
    :param backs: if 'none' is choosed as aggr_options, this is the amount of events back taken
    :param file: the file name
    :param repo_holder: the repo holder
    :param cur_repo: the current repo
    :param cur_list: the current list of events
    :param tag: the tag to add to the window
    :return: None
    """
    for cur in cur_list:
        res = get_event_window(cur_repo, cur, aggr_options,
                               days=days, hours=hours, backs=backs, resample=resample)
        details = (file, cur, tag)
        window_lst.append(res)
        details_lst.append(details)


def fix_repo_shape(all_set, cur_repo):
    cur_repo['created_at'] = pd.to_datetime(cur_repo['created_at'], utc=True)
    cur_repo = cur_repo[~cur_repo.duplicated(subset=['created_at','Vuln'],keep='first')]
    cur_repo = cur_repo.set_index(["created_at"])
    cur_repo = cur_repo.sort_index()
    cur_repo = cur_repo[cur_repo.index.notnull()]
    all_set.update(cur_repo.type.unique())
    cur_repo['idx'] = range(len(cur_repo))
    cur_repo = cur_repo.reset_index().set_index(["created_at", "idx"])

    # cur_repo = cur_repo[cur_repo.index.notnull()]
    for commit_change in ["Add", "Del", "Files"]:
        cur_repo[commit_change].fillna(0, inplace=True)
        cur_repo[commit_change] = cur_repo[commit_change].astype(int)
        cur_repo[commit_change] = (
            cur_repo[commit_change] - cur_repo[commit_change].mean()) / cur_repo[commit_change].std()

    cur_repo = add_type_one_hot_encoding(cur_repo)
    cur_repo = cur_repo.drop(["type"], axis=1)
    cur_repo = cur_repo.drop(["name"], axis=1)
    cur_repo = cur_repo.drop(["Unnamed: 0"], axis=1)
    cur_repo = cur_repo.drop(["Hash"], axis=1)
    return cur_repo


def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample):
    """
    :return: name of the directory to save the data in
    """
    name_template = f"{str(aggr_options)}_{benign_vuln_ratio}_H{hours}_D{days}_R{resample}_B{backs}"
    logging.debug(name_template)
    return name_template


def extract_dataset(aggr_options=Aggregate.none, benign_vuln_ratio=1, hours=0, days=10, resample=12, backs=50,
                    cache=False):
    """
    :param aggr_options: Aggregate.none, Aggregate.before_cve, Aggregate.after_cve
    :param benign_vuln_ratio: ratio of benign to vuln events
    :param hours: hours before and after vuln event
    :param days: days before and after vuln event
    :param resample: resample window
    :param backs: number of backs to use
    :param cache: if true, will use cached data
    :return: a list of Repository objects and dir name
    """

    dirname = make_new_dir_name(
        aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    if cache and os.path.isdir("ready_data/" + dirname) and len(os.listdir("ready_data/" + dirname)) != 0:
        logging.info(f"Loading Dataset {dirname}")
        all_repos = []
        for file in os.listdir("ready_data/" + dirname):
            with open("ready_data/" + dirname + "/" + file, 'rb') as f:
                repo = pickle.load(f)
                all_repos.append(repo)

    else:
        logging.info(f"Creating Dataset {dirname}")
        all_repos = create_dataset(
            aggr_options, benign_vuln_ratio, hours, days, resample, backs)

    return all_repos, dirname


def evaluate_data(X_train, y_train, X_val, y_val, exp_name, epochs=20):
    """
    Evaluate the model with the given data.
    """

    model1 = conv1d(X_train.shape[1], X_train.shape[2])

    # define model
    model2 = conv1d_more_layers(X_train.shape[1], X_train.shape[2])

    model3 = lstm(X_train.shape[1], X_train.shape[2])

    model4 = small_conv1d(X_train.shape[1], X_train.shape[2])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)

    verbose = 0
    if logging.INFO <= logging.root.level:
        verbose = 1

    model = model1
    history = model.fit(X_train, y_train, verbose=verbose, epochs=epochs, shuffle=True,
                        validation_data=(X_val, y_val), callbacks=[])

    model_name = f'{model=}'.split('=')[0]
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')
    pyplot.draw()

    safe_mkdir("figs")
    pyplot.savefig(f"figs/{exp_name}_{epochs}_{model_name}.png")

    # Final evaluation of the model
    return model


def acquire_commits(name, date, ignore_errors=False):
    """
    Acquire the commits for the given repository.
    """
    group, repo = name.replace(".csv", "").split("_", 1)

    github_format = "%Y-%m-%dT00:00:00"
    for branch in ["master", "main"]:
        res = helper.run_query(
            helper.commits_between_dates.format(group,
                                                repo,
                                                branch,
                                                date.strftime(github_format),
                                                (date + datetime.timedelta(days=1)
                                                 ).strftime(github_format)
                                                ), ignore_errors=ignore_errors)
        if "data" in res:
            if "repository" in res["data"]:
                if "object" in res['data']['repository']:
                    obj = res['data']['repository']['object']
                    if obj is None:
                        continue
                    if "history" in obj:
                        return res['data']['repository']['object']['history']['nodes']
    return ""


def check_results(X_test, y_test, pred, model, exp_name):
    """
    Check the results of the model.
    """
    used_y_test = np.asarray(y_test).astype('float32')
    scores = model.evaluate(X_test, used_y_test, verbose=0)
    model_name = f'{model=}'.split('=')[0]
    max_f1, thresh, _ = find_best_f1(X_test, used_y_test, model)
    logging.debug(max_f1, thresh)
    with open(f"results/{exp_name}_{model_name}.txt", 'w') as mfile:
        mfile.write('Accuracy: %.2f%%\n' % (scores[1] * 100))
        mfile.write('fscore: %.2f%%\n' % (max_f1 * 100))

        logging.critical('Accuracy: %.2f%%' % (scores[1] * 100))
        logging.critical('fscore: %.2f%%' % (max_f1 * 100))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(used_y_test, pred)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2

    plt.plot(fpr['micro'], tpr['micro'], color="darkorange", lw=lw,
             label="ROC curve (area = %0.2f)" % roc_auc['micro'])
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(f"figs/auc_{exp_name}_{roc_auc['micro']}.png")

    return scores[1]


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hours', type=int, default=0, help='hours back')
    parser.add_argument('-d', '--days', type=int, default=10, help='days back')
    parser.add_argument('--resample', type=int, default=24,
                        help='number of hours that should resample aggregate')
    parser.add_argument('-r', '--ratio', type=int,
                        default=1, help='benign vuln ratio')
    parser.add_argument('-a', '--aggr', type=Aggregate,
                        action=EnumAction, default=Aggregate.none)
    parser.add_argument('-b', '--backs', type=int, default=10,
                        help=' using none aggregation, operations back')
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('-c', '--cache', '--cached', help="Use Cached Data",
                        action="store_const", dest="cache",  const=True)
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help=' using none aggregation, operations back')
    parser.add_argument('-f', '--find-fp', help="Find False positive commits", action="store_const",
                        dest="fp", const=True)

    args = parser.parse_args()
    return args


def split_into_x_and_y(repos, with_details=False):
    """
    Split the repos into X and Y.
    """
    X_train, y_train = [], []
    details = []
    for repo in repos:
        x, y = repo.get_all_lst()
        if with_details:
            details.append(repo.get_all_details())
        X_train.append(x)
        y_train.append(y)
    if X_train:
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

    if with_details:
        if details:
            details = np.concatenate(details)
        return X_train, y_train, details

    return X_train, y_train


def init():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = '0'


def main():
    args = parse_args()
    logging.basicConfig(level=args.loglevel)

    init()

    all_repos, exp_name = extract_dataset(aggr_options=args.aggr,
                                          resample=args.resample,
                                          benign_vuln_ratio=args.ratio,
                                          hours=args.hours,
                                          days=args.days,
                                          backs=args.backs,
                                          cache=args.cache)

    to_pad = 0
    num_of_vulns = 0
    for repo in all_repos:
        num_of_vulns += repo.get_num_of_vuln()
        if len(repo.get_all_lst()[0].shape) > 1:
            to_pad = max(to_pad, repo.get_all_lst()[0].shape[1])
        else:
            all_repos.remove(repo)

    for repo in all_repos:
        repo.pad_repo(to_pad=to_pad)
    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.10
    train_size = int(TRAIN_SIZE * num_of_vulns)
    validation_size = int(VALIDATION_SIZE * num_of_vulns)
    test_size = num_of_vulns - train_size - validation_size

    logging.info(f"Train size: {train_size}")
    logging.info(f"Validation size: {validation_size}")
    logging.info(f"Test size: {test_size}")

    train_repos = []
    validation_repos = []
    test_repos = []

    vuln_counter = 0
    random.shuffle(all_repos)
    for repo in all_repos:
        if(vuln_counter < validation_size):
            logging.debug(f"Train - {repo.file}")
            validation_repos.append(repo)
        elif(vuln_counter < train_size + validation_size):
            logging.debug(f"Val - {repo.file}")
            train_repos.append(repo)
        else:
            logging.debug(f"Test - {repo.file}")
            test_repos.append(repo)
        vuln_counter += repo.get_num_of_vuln()

    if not train_repos or not validation_repos or not test_repos:
        raise Exception("Not enough data, or data not splitted well")

    X_train, y_train = split_into_x_and_y(train_repos)
    X_val, y_val, val_details = split_into_x_and_y(
        validation_repos, with_details=True)
    X_test, y_test, test_details = split_into_x_and_y(
        test_repos, with_details=True)

    model = evaluate_data(X_train, y_train, X_val, y_val,
                          exp_name, epochs=args.epochs)

    pred = model.predict(X_test).reshape(-1)

    _ = check_results(X_test, y_test, pred, model, exp_name)

    if args.fp:
        safe_mkdir("output")
        safe_mkdir(f"output/{exp_name}")

        summary_md = ""
        summary_md += f"# {exp_name}\n"

        df = DataFrame(zip(y_test, pred, test_details[:, 0], test_details[:, 1]), columns=[
                       'real', 'pred', 'file', 'timestamp'])

        fps = df[((df['pred'] > df['real']) & (df['pred'] > 0.5))]

        groups = fps.groupby('file')
        for name, group in groups:
            summary_md += f"## {name}\n"
            summary_md += "\n".join(
                list(group['timestamp'].apply(lambda x: "* " + str(x[0]))))
            summary_md += "\n"

        with open(f"output/{exp_name}/summary.md", "w") as f:
            f.write(summary_md)

        for _, row in tqdm.tqdm(list(fps.iterrows())):
            if "tensorflow" in row["file"]:
                logging.debug("Skipping over tf")

                continue
            commits = acquire_commits(
                row["file"], row["timestamp"][0], ignore_errors=True)
            if commits:
                with open(f'output/{exp_name}/{row["file"]}_{row["timestamp"][0].strftime("%Y-%m-%d")}.json',
                          'w+') as mfile:
                    json.dump(commits, mfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
