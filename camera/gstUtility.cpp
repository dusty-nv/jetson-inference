/*
 * inference-101
 */

#include "gstUtility.h"

#include <gst/gst.h>
#include <stdint.h>
#include <stdio.h>


inline const char* gst_debug_level_str( GstDebugLevel level )
{
	switch (level)
	{
		case GST_LEVEL_NONE:	return "GST_LEVEL_NONE   ";
		case GST_LEVEL_ERROR:	return "GST_LEVEL_ERROR  ";
		case GST_LEVEL_WARNING:	return "GST_LEVEL_WARNING";
		case GST_LEVEL_INFO:	return "GST_LEVEL_INFO   ";
		case GST_LEVEL_DEBUG:	return "GST_LEVEL_DEBUG  ";
		case GST_LEVEL_LOG:		return "GST_LEVEL_LOG    ";
		case GST_LEVEL_FIXME:	return "GST_LEVEL_FIXME  ";
#ifdef GST_LEVEL_TRACE
		case GST_LEVEL_TRACE:	return "GST_LEVEL_TRACE  ";
#endif
		case GST_LEVEL_MEMDUMP:	return "GST_LEVEL_MEMDUMP";
    		default:				return "<unknown>        ";
    }
}

#define SEP "              "

void rilog_debug_function(GstDebugCategory* category, GstDebugLevel level,
                          const gchar* file, const char* function,
                          gint line, GObject* object, GstDebugMessage* message,
                          gpointer data)
{
	if( level > GST_LEVEL_WARNING /*GST_LEVEL_INFO*/ )
		return;

	//gchar* name = NULL;
	//if( object != NULL )
	//	g_object_get(object, "name", &name, NULL);

	const char* typeName  = " ";
	const char* className = " ";

	if( object != NULL )
	{
		typeName  = G_OBJECT_TYPE_NAME(object);
		className = G_OBJECT_CLASS_NAME(object);
	}

	printf(LOG_GSTREAMER "%s %s %s\n" SEP "%s:%i  %s\n" SEP "%s\n", 
		  	gst_debug_level_str(level), typeName,
		  	gst_debug_category_get_name(category), file, line, function, 
            	gst_debug_message_get(message));

}


bool gstreamerInit()
{
	int argc = 0;
	//char* argv[] = { "none" };

	if( !gst_init_check(&argc, NULL, NULL) )
	{
		printf(LOG_GSTREAMER "failed to initialize gstreamer library with gst_init()\n");
		return false;
	}

	uint32_t ver[] = { 0, 0, 0, 0 };
	gst_version( &ver[0], &ver[1], &ver[2], &ver[3] );

	printf(LOG_GSTREAMER "initialized gstreamer, version %u.%u.%u.%u\n", ver[0], ver[1], ver[2], ver[3]);


	// debugging
	gst_debug_remove_log_function(gst_debug_log_default);
	
	if( true )
	{
		gst_debug_add_log_function(rilog_debug_function, NULL, NULL);

		gst_debug_set_active(true);
		gst_debug_set_colored(false);
	}
	
	return true;
}
//---------------------------------------------------------------------------------------------

static void gst_print_one_tag(const GstTagList * list, const gchar * tag, gpointer user_data)
{
  int i, num;

  num = gst_tag_list_get_tag_size (list, tag);
  for (i = 0; i < num; ++i) {
    const GValue *val;

    /* Note: when looking for specific tags, use the gst_tag_list_get_xyz() API,
     * we only use the GValue approach here because it is more generic */
    val = gst_tag_list_get_value_index (list, tag, i);
    if (G_VALUE_HOLDS_STRING (val)) {
      printf("\t%20s : %s\n", tag, g_value_get_string (val));
    } else if (G_VALUE_HOLDS_UINT (val)) {
      printf("\t%20s : %u\n", tag, g_value_get_uint (val));
    } else if (G_VALUE_HOLDS_DOUBLE (val)) {
      printf("\t%20s : %g\n", tag, g_value_get_double (val));
    } else if (G_VALUE_HOLDS_BOOLEAN (val)) {
      printf("\t%20s : %s\n", tag,
          (g_value_get_boolean (val)) ? "true" : "false");
    } else if (GST_VALUE_HOLDS_BUFFER (val)) {
      //GstBuffer *buf = gst_value_get_buffer (val);
      //guint buffer_size = GST_BUFFER_SIZE(buf);

      printf("\t%20s : buffer of size %u\n", tag, /*buffer_size*/0);
    } /*else if (GST_VALUE_HOLDS_DATE_TIME (val)) {
      GstDateTime *dt = (GstDateTime*)g_value_get_boxed (val);
      gchar *dt_str = gst_date_time_to_iso8601_string (dt);

      printf("\t%20s : %s\n", tag, dt_str);
      g_free (dt_str);
    }*/ else {
      printf("\t%20s : tag of type '%s'\n", tag, G_VALUE_TYPE_NAME (val));
    }
  }
}

static const char* gst_stream_status_string( GstStreamStatusType status )
{
	switch(status)
	{
		case GST_STREAM_STATUS_TYPE_CREATE:	return "CREATE";
		case GST_STREAM_STATUS_TYPE_ENTER:		return "ENTER";
		case GST_STREAM_STATUS_TYPE_LEAVE:		return "LEAVE";
		case GST_STREAM_STATUS_TYPE_DESTROY:	return "DESTROY";
		case GST_STREAM_STATUS_TYPE_START:		return "START";
		case GST_STREAM_STATUS_TYPE_PAUSE:		return "PAUSE";
		case GST_STREAM_STATUS_TYPE_STOP:		return "STOP";
		default:							return "UNKNOWN";
	}
}

// gst_message_print
gboolean gst_message_print(GstBus* bus, GstMessage* message, gpointer user_data)
{

	switch (GST_MESSAGE_TYPE (message)) 
	{
		case GST_MESSAGE_ERROR: 
		{
			GError *err = NULL;
			gchar *dbg_info = NULL;
 
			gst_message_parse_error (message, &err, &dbg_info);
			printf(LOG_GSTREAMER "gstreamer %s ERROR %s\n", GST_OBJECT_NAME (message->src), err->message);
        		printf(LOG_GSTREAMER "gstreamer Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
        
			g_error_free(err);
        		g_free(dbg_info);
			//g_main_loop_quit (app->loop);
        		break;
		}
		case GST_MESSAGE_EOS:
		{
			printf(LOG_GSTREAMER "gstreamer %s recieved EOS signal...\n", GST_OBJECT_NAME(message->src));
			//g_main_loop_quit (app->loop);		// TODO trigger plugin Close() upon error
			break;
		}
		case GST_MESSAGE_STATE_CHANGED:
		{
			GstState old_state, new_state;
    
			gst_message_parse_state_changed(message, &old_state, &new_state, NULL);
			
			printf(LOG_GSTREAMER "gstreamer changed state from %s to %s ==> %s\n",
							gst_element_state_get_name(old_state),
							gst_element_state_get_name(new_state),
						     GST_OBJECT_NAME(message->src));
			break;
		}
		case GST_MESSAGE_STREAM_STATUS:
		{
			GstStreamStatusType streamStatus;
			gst_message_parse_stream_status(message, &streamStatus, NULL);
			
			printf(LOG_GSTREAMER "gstreamer stream status %s ==> %s\n",
							gst_stream_status_string(streamStatus), 
							GST_OBJECT_NAME(message->src));
			break;
		}
		case GST_MESSAGE_TAG: 
		{
			GstTagList *tags = NULL;

			gst_message_parse_tag(message, &tags);

#ifdef gst_tag_list_to_string
			gchar* txt = gst_tag_list_to_string(tags);
#else
			gchar* txt = "missing gst_tag_list_to_string()";
#endif

			printf(LOG_GSTREAMER "gstreamer %s %s\n", GST_OBJECT_NAME(message->src), txt);

			g_free(txt);			
			//gst_tag_list_foreach(tags, gst_print_one_tag, NULL);
			gst_tag_list_free(tags);
			break;
		}
		default:
		{
			printf(LOG_GSTREAMER "gstreamer msg %s ==> %s\n", gst_message_type_get_name(GST_MESSAGE_TYPE(message)), GST_OBJECT_NAME(message->src));
			break;
		}
	}

	return TRUE;
}

